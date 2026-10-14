from __future__ import annotations

from typing import Protocol


class UnifiedBackboneAdapter(Protocol):
    """Minimal contract used by the Scaffold-GS unified runner.

    Future MLP 3DGS adapters should hide backbone-specific scene loading,
    renderer calls, checkpoint slice loading, and plug-in tensor layout behind
    these methods.
    """

    name: str
    supports_condition_injection: bool
    condition_injection_note: str | None

    def load_scene(self, dataset, iteration: int = -1): ...

    def set_plugin_layout(self, gaussians, layout: dict[str, tuple[int, int]]) -> None: ...

    def validate_inject_dim(self, gaussians, requested_inject_dim: int) -> dict[str, int]: ...

    def zero_disabled_plugin_slices(self, gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None: ...

    def register_disabled_plugin_slice_hooks(self, gaussians, layout: dict[str, tuple[int, int]], disabled: set[str]) -> None: ...

    def load_plugin_slice(self, gaussians, dataset, name: str, plugin_path: str, target_slice: tuple[int, int], logger, iteration: int = -1) -> bool: ...

    def require_unified_plugin_checkpoint(self, model_path, iteration: int, total_dim: int) -> None: ...

    def render(self, camera, gaussians, pipe, background): ...

    def render_baseline(self, camera, gaussians, pipe, background): ...

    def trainable_parameter_count(self, gaussians) -> int: ...
