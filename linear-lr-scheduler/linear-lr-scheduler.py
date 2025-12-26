def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    if step >= total_steps:
        return float(final_lr)

    # No warmup case
    if warmup_steps == 0:
        return float(
            initial_lr +
            (final_lr - initial_lr) * step / total_steps
        )

    # Warmup phase
    if step < warmup_steps:
        return float(initial_lr * step / warmup_steps)

    # Decay phase
    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps

    return float(
        initial_lr +
        (final_lr - initial_lr) * decay_step / decay_steps
    )