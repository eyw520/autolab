import torch

from pretraining_gpt.experiment.model.gpt import GPT


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    lr_t: torch.Tensor,
    beta1_t: torch.Tensor,
    beta2_t: torch.Tensor,
    eps_t: torch.Tensor,
    wd_t: torch.Tensor,
) -> None:
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(
    stacked_grads: torch.Tensor,
    stacked_params: torch.Tensor,
    momentum_buffer: torch.Tensor,
    second_momentum_buffer: torch.Tensor,
    momentum_t: torch.Tensor,
    lr_t: torch.Tensor,
    wd_t: torch.Tensor,
    beta2_t: torch.Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


_compiled_adamw = None
_compiled_muon = None


def get_compiled_steps(use_cuda: bool):
    global _compiled_adamw, _compiled_muon
    if use_cuda and _compiled_adamw is None:
        _compiled_adamw = torch.compile(adamw_step_fused, dynamic=False, fullgraph=True)
        _compiled_muon = torch.compile(muon_step_fused, dynamic=False, fullgraph=True)
    return (_compiled_adamw or adamw_step_fused, _compiled_muon or muon_step_fused)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict], use_cuda: bool = False):
        super().__init__(param_groups, defaults={})
        self.use_cuda = use_cuda
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        adamw_fn, _ = get_compiled_steps(self.use_cuda)
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_fn(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        _, muon_fn = get_compiled_steps(self.use_cuda)
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_fn(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


def setup_optimizer(
    model: GPT,
    unembedding_lr: float = 0.004,
    embedding_lr: float = 0.2,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
    adam_betas: tuple[float, float] = (0.8, 0.95),
    scalar_lr: float = 0.5,
    use_cuda: bool = False,
) -> MuonAdamW:
    model_dim = model.config.n_embd
    matrix_params = list(model.transformer.h.parameters())
    value_embeds_params = list(model.value_embeds.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]

    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

    adamw_defaults = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
    param_groups = [
        dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, **adamw_defaults),
        dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, **adamw_defaults),
        dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, **adamw_defaults),
        dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, **adamw_defaults),
        dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(
            dict(
                kind="muon",
                params=group_params,
                lr=matrix_lr,
                momentum=0.95,
                ns_steps=5,
                beta2=0.95,
                weight_decay=weight_decay,
            )
        )

    optimizer = MuonAdamW(param_groups, use_cuda=use_cuda)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
