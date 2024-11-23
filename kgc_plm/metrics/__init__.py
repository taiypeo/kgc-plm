from ..graphs import BaseGraph

HITS_AT_K: list[int] = [1, 3, 10]


def _process_triplets(
    items: dict[str, list[str]],
    ranking: dict[tuple[str, str], list[str]],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {
        f"hits@{k}": []
        for k in HITS_AT_K
    }
    result["mrr"] = []

    for head, relation, tail in zip(items["head"], items["relation"], items["tail"]):
        ranked_entity_ids = ranking[(head, relation)]
        try:
            rank = ranked_entity_ids.index(tail) + 1
            result["mrr"].append(1. / rank)
            for k in HITS_AT_K:
                result[f"hits@{k}"].append(float(rank <= k))
        except ValueError:
            result["mrr"].append(0.)
            for k in HITS_AT_K:
                result[f"hits@{k}"].append(0.)

    return result


def calculate_metrics(
    ranking: dict[tuple[str, str], list[str]],
    graph: BaseGraph,
    split: str = "test",
    batch_size: int = 1000,
) -> dict[str, float]:
    mapped = graph.triplets[split].map(
        lambda items: _process_triplets(items, ranking),
        batched=True,
        batch_size=batch_size
    )
    result = {f"hits@{k}": sum(mapped[f"hits@{k}"]) / len(mapped) for k in HITS_AT_K}
    result["mrr"] = sum(mapped["mrr"]) / len(mapped)
    return result
