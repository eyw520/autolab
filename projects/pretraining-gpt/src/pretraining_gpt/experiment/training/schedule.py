def get_lr_multiplier(
    progress: float,
    warmup_ratio: float = 0.0,
    warmdown_ratio: float = 0.5,
    final_lr_frac: float = 0.0,
) -> float:
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float, base_weight_decay: float) -> float:
    return base_weight_decay * (1 - progress)
