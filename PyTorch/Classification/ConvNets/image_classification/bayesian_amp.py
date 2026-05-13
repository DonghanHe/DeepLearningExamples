import torch
import torch.nn as nn
from typing import Dict, Optional

# FP8 is excluded: standard ATen mm/conv ops require torch._scaled_mm for FP8 inputs.
# Thresholds map log2(SNR) → the dtype that can represent that SNR level.
# Ordered lowest-to-highest; the first match wins.
_SNR_THRESHOLDS = [
    (17.58, torch.bfloat16),
    (23.58, torch.float16),
    # > 23.58 → torch.float32 (default)
]


class BayesianAMPHook:
    """
    Per-module precision router based on the Vadam posterior SNR.

    Design: Bayesian AMP runs *inside* a global torch.autocast(dtype=base_dtype)
    context (typically BF16 on A100). For layers whose SNR fits within
    base_dtype's capacity, NO hooks are registered — zero Python call overhead.
    Hooks are only dynamically attached when update_dtype() determines a layer
    needs to deviate (upgrade to FP32), and detached again if it later drops back.

    This means the common case (all layers BF16 early in training) has identical
    overhead to standard torch.autocast, since there are no hooks at all.
    """

    def __init__(
        self,
        module: nn.Module,
        param: nn.Parameter,
        optimizer: torch.optim.Optimizer,
        N: int,
        beta2: float = 0.999,
        base_dtype: torch.dtype = torch.bfloat16,
    ):
        self.module = module
        self.param = param
        self.opt = optimizer
        self.N = N
        self.beta2 = beta2
        self.base_dtype = base_dtype
        self._dtype = base_dtype
        self._step = 0
        self._ctx: Optional[torch.autocast] = None
        # Start with no hooks registered — attached dynamically when needed
        self._pre: Optional[torch.utils.hooks.RemovableHook] = None
        self._post: Optional[torch.utils.hooks.RemovableHook] = None

    def _set_hooks(self, active: bool) -> None:
        """Register hooks when a layer deviates from base_dtype; remove when it returns."""
        if active and self._pre is None:
            self._pre = self.module.register_forward_pre_hook(self._pre_hook)
            self._post = self.module.register_forward_hook(self._post_hook)
        elif not active and self._pre is not None:
            self._pre.remove()
            self._post.remove()
            self._pre = None
            self._post = None

    def update_dtype(self):
        """Recompute target dtype from current optimizer state. Call after opt.step()."""
        state = self.opt.state.get(self.param)
        if not state:
            return
        v_t = state.get("exp_avg_sq")
        if v_t is None:
            return

        self._step += 1
        # Bias correction: v_t initialized at 0 makes early SNR artificially high
        # without this, causing premature precision decisions.
        bias_corr = 1.0 - self.beta2 ** self._step
        v_hat = v_t / bias_corr

        wd = 0.0
        for g in self.opt.param_groups:
            if any(p is self.param for p in g["params"]):
                wd = g.get("weight_decay", 0.0)
                break

        with torch.no_grad():
            # 10th-percentile SNR: weakest-link proxy consistent with §2.
            snr = torch.abs(self.param) * torch.sqrt(self.N * v_hat + max(wd, 1e-8))
            log2_snr = torch.log2(torch.quantile(snr.float(), 0.10) + 1e-8).item()

        new_dtype = self.base_dtype
        for threshold, dtype in _SNR_THRESHOLDS:
            if log2_snr <= threshold:
                new_dtype = dtype
                break
        if log2_snr > 23.58:
            new_dtype = torch.float32

        self._dtype = new_dtype
        # Hooks are only needed when this layer deviates from the global base dtype
        self._set_hooks(new_dtype != self.base_dtype)

    def _pre_hook(self, module, args):
        device_type = "cuda" if self.param.is_cuda else "cpu"
        if self._dtype == torch.float32:
            self._ctx = torch.amp.autocast(device_type=device_type, enabled=False)
        else:
            self._ctx = torch.amp.autocast(device_type=device_type, dtype=self._dtype)
        self._ctx.__enter__()

    def _post_hook(self, module, args, output):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            if isinstance(output, torch.Tensor) and output.dtype != self.base_dtype:
                return output.to(self.base_dtype)

    def remove(self):
        self._set_hooks(False)


class BayesianAMPManager:
    """
    Attaches a BayesianAMPHook to every nn.Linear and nn.Conv2d whose weight
    is tracked by the given optimizer. Handles DDP-wrapped models transparently.

    Must be used inside a global torch.amp.autocast(dtype=base_dtype) context.
    Hooks are no-ops for layers whose SNR fits within base_dtype; they only
    intervene to upgrade high-SNR layers to FP32.

    Usage:
        mgr = BayesianAMPManager(model, optimizer, N=len(dataset))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        mgr.step()   # refresh precision decisions
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        N: int,
        base_dtype: torch.dtype = torch.bfloat16,
    ):
        self.base_dtype = base_dtype
        inner = getattr(model, "module", model)  # unwrap DistributedDataParallel
        param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        self._hooks: Dict[str, BayesianAMPHook] = {}
        for name, mod in inner.named_modules():
            if not isinstance(mod, (nn.Linear, nn.Conv2d)):
                continue
            if id(mod.weight) in param_ids:
                self._hooks[name] = BayesianAMPHook(
                    mod, mod.weight, optimizer, N, base_dtype=base_dtype
                )

    def step(self):
        """Update per-layer precision decisions. Call once per optimizer.step()."""
        for h in self._hooks.values():
            h.update_dtype()

    def dtype_summary(self) -> Dict[str, str]:
        """Return {layer_name: dtype_str} for inspection/logging."""
        return {name: str(h._dtype) for name, h in self._hooks.items()}

    def remove(self):
        """Detach all hooks (e.g., before switching to eval-only mode)."""
        for h in self._hooks.values():
            h.remove()
        self._hooks.clear()
