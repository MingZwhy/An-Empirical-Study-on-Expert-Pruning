"""Local expert-pruning hooks for vLLM MoE routing."""

from expert_pruning.routing import configure_expert_router
from expert_pruning.vllm_patch import (
    install_router_distribution_observer,
    install_vllm_expert_pruning_router,
)

__all__ = [
    "configure_expert_router",
    "install_router_distribution_observer",
    "install_vllm_expert_pruning_router",
]
