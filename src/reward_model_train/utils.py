"""Shared utilities for the reward model training framework."""

from __future__ import annotations

import torch.distributed as dist


def dist_debug() -> None:
    """Drop into a debugger on rank 0, then sync all ranks."""
    if dist.get_rank() == 0:
        breakpoint()
    dist.barrier()
