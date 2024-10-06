from .base import BaseGraph
from .fb15k_237 import FB15K_237


def get_graph(
    graph_name: str,
    batch_size: int,
    cache_dir: str,
    **kwargs,
) -> BaseGraph:
    if graph_name == "fb15k_237":
        return FB15K_237(
            batch_size=batch_size,
            cache_dir=cache_dir,
            **kwargs,
        )

    raise ValueError(f"Unknown graph: {graph_name}")
