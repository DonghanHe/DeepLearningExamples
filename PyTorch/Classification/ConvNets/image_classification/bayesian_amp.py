import torch
import torch.nn as nn
from typing import Dict, Optional

# FP8 is excluded: standard ATen mm/conv ops require torch._scaled_mm for FP8 inputs.
# Thresholds are ordered lowest-to-highest; the first match wins.
_SNR_THRESHOLDS = [
    (17.58, torch.bfloat16),
    (23.58, torch.float16),
]


class BayesianAMPHook:
    """
    Per-module precision router based on the Vadam posterior SNR.

    Attaches pre- and post-forward hooks to a single nn.Linear or nn.Conv2d.
    The pre-hook opens a nested torch.autocast context at the dynamically
    chosen dtype; the post-hook closes it and upcasts the output back to FP32
    to protect the residual backbone.

    Call update_dtype() once per optimizer.step() to refresh the precision
    decision from the bias-corrected AdamW second moment.
    """

    def __init__(
        self,
        module: nn.Module,
        param: nn.Parameter,
        optimizer: torch.optim.Optimizer,
        N: int,
        beta2: float = 0.999,
    ):
        self.param = param
        self.opt = optimizer
        self.N = N
        self.beta2 = beta2
        self._dtype = torch.float32
        self._step = 0
        self._ctx: Optional[torch.autocast] = None
        self._pre = module.register_forward_pre_hook(self._pre_hook)
        self._post = module.register_forward_hook(self._post_hook)

    def update_dtype(self):
        """Recompute target dtype from current optimizer state. Call after opt.step()."""
        state = self.opt.state.get(self.param)
        if not state:
            return
        v_t = state.get("exp_avg_sq")
        if v_t is None:
            return

        self._step += 1
        # Bias correction: v_t is initialized to 0, so early steps are downward-biased.
        # Without this, SNR is artificially high early in training, causing premature downcast.
        bias_corr = 1.0 - self.beta2 ** self._step
        v_hat = v_t / bias_corr

        wd = 0.0
        for g in self.opt.param_groups:
            if any(p is self.param for p in g["params"]):
                wd = g.get("weight_decay", 0.0)
                break

        with torch.no_grad():
            # SNR per element; use 10th-percentile as the weakest-link proxy
            # (consistent with §2 of the Bayesian AMP derivation).
            snr = torch.abs(self.param) * torch.sqrt(self.N * v_hat + max(wd, 1e-8))
            log2_snr = torch.log2(torch.quantile(snr.float(), 0.10) + 1e-8).item()

        self._dtype = torch.float32
        for threshold, dtype in _SNR_THRESHOLDS:
            if log2_snr <= threshold:
                self._dtype = dtype
                break

    def _pre_hook(self, module, args):
        if self._dtype != torch.float32:
            device_type = "cuda" if self.param.is_cuda else "cpu"
            self._ctx = torch.autocast(device_type=device_type, dtype=self._dtype)
            self._ctx.__enter__()

    def _post_hook(self, module, args, output):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            if isinstance(output, torch.Tensor) and output.dtype != torch.float32:
                return output.to(torch.float32)

    def remove(self):
        self._pre.remove()
        self._post.remove()


class BayesianAMPManager:
    """
    Attaches a BayesianAMPHook to every nn.Linear and nn.Conv2d whose weight
    is tracked by the given optimizer. Handles DDP-wrapped models transparently.

    Usage:
        mgr = BayesianAMPManager(model, optimizer, N=len(dataset))
        # inside train loop, after optimizer.step():
        mgr.step()
        # at end of training:
        mgr.remove()
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, N: int):
        inner = getattr(model, "module", model)  # unwrap DistributedDataParallel
        param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        self._hooks: Dict[str, BayesianAMPHook] = {}
        for name, mod in inner.named_modules():
            if not isinstance(mod, (nn.Linear, nn.Conv2d)):
                continue
            if id(mod.weight) in param_ids:
                self._hooks[name] = BayesianAMPHook(mod, mod.weight, optimizer, N)

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
