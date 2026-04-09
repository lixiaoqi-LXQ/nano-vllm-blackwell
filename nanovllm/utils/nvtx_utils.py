import torch
import torch.nn as nn

_SKIPPED_MODULES = (nn.Identity, nn.Dropout)


def _nvtx_range_push(name):
    def hook(module, args):
        torch.cuda.nvtx.range_push(name)
    return hook


def _nvtx_range_pop():
    def hook(module, args, output):
        torch.cuda.nvtx.range_pop()
    return hook


def register_nvtx_hooks(model: nn.Module):
    """Register NVTX range_push/pop hooks on all model modules."""
    for name, module in model.named_modules(prefix=model.__class__.__name__):
        if isinstance(module, _SKIPPED_MODULES):
            continue
        module.register_forward_pre_hook(_nvtx_range_push(name))
        module.register_forward_hook(_nvtx_range_pop())
