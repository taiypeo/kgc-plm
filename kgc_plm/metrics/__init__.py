from ..graphs import BaseGraph


def calculate_metrics(
    ranking: dict[tuple[str, str], list[str]],
    graph: BaseGraph,
    split: str = "test",
    hits_at_k: list[int] = [1, 3, 10],
) -> dict[str, float]:
    max_hits = max(hits_at_k)

    hits = [0.0] * range(max_hits)
    sum_reciprocal_ranks = 0.0
    for triplet in graph.triplets[split]:
        ranked_entity_ids = ranking[(triplet["head"], triplet["relation"])]
        try:
            rank = ranked_entity_ids.index(triplet["tail"])
            sum_reciprocal_ranks += 1.0 / rank
            for i in range(1, len(hits) + 1):
                hits[i - 1] += float(rank <= i)
        except ValueError:
            continue  # reciprocal rank is 0 in this case

    result = {f"hits@{k}": hits[k - 1] / len(graph.triplets[split]) for k in hits_at_k}
    result["mrr"] = sum_reciprocal_ranks / len(graph.triplets[split])
    return result
