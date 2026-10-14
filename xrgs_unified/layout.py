from __future__ import annotations


PluginLayout = dict[str, tuple[int, int]]


def build_plugin_layout(ss_dim: int, display_dim: int, lens_dim: int) -> PluginLayout:
    layout: PluginLayout = {}
    start = 0
    for name, dim in (("ss", ss_dim), ("display", display_dim), ("lens", lens_dim)):
        if dim < 0:
            raise ValueError(f"{name} plug-in dimension must be non-negative, got {dim}")
        if dim > 0:
            layout[name] = (start, start + dim)
            start += dim
    return layout


def layout_to_metadata(layout: PluginLayout) -> dict[str, dict[str, int]]:
    return {
        name: {"start": int(start), "end": int(end), "dim": int(end - start)}
        for name, (start, end) in layout.items()
    }


def total_layout_dim(layout: PluginLayout) -> int:
    return sum(end - start for start, end in layout.values())


def disabled_plugin_modules(args, layout: PluginLayout) -> set[str]:
    disabled = set()
    if getattr(args, "disable_ss", False):
        disabled.add("ss")
    if getattr(args, "disable_display", False):
        disabled.add("display")
    if getattr(args, "disable_lens", False):
        disabled.add("lens")
    return {name for name in disabled if name in layout}
