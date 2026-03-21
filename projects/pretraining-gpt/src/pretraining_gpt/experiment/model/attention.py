import torch
import torch.nn as nn
import torch.nn.functional as F


fa3 = None
_fa3_initialized = False


def init_flash_attention():
    global fa3, _fa3_initialized
    if _fa3_initialized:
        return fa3
    _fa3_initialized = True

    if torch.cuda.is_available():
        try:
            from kernels import get_kernel

            cap = torch.cuda.get_device_capability()
            repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
            fa3 = get_kernel(repo).flash_attn_interface
        except ImportError:
            pass
    return fa3


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, n_layer: int, layer_idx: int):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, n_kv_head, bias=False) if has_ve(layer_idx, n_layer) else None

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        fa3_module = init_flash_attention()
        if fa3_module is not None:
            y = fa3_module.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.n_kv_head < self.n_head:
                k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, n_layer: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, n_layer, layer_idx)
        self.mlp = MLP(n_embd)

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x
