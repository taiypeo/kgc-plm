import random

from datasets import Dataset, DatasetDict

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


def _get_triplets(
    triplets_dataset: Dataset,
    entities: list[str],
    relations: list[str],
    should_shuffle: bool = False,
    size: float = 1.,
    random_seed: int = 42,
    valid_triplets: set | None = None,
    test_triplets: set | None = None,
) -> dict[bool, set[tuple[str, str, str]]]:
    pos_triplets_dataset = triplets_dataset
    if should_shuffle:
        pos_triplets_dataset = pos_triplets_dataset.shuffle(seed=random_seed)

    pos_triplets = set()
    i = 0
    while i < len(pos_triplets_dataset) and len(pos_triplets) < int(size * len(pos_triplets_dataset)):
        new_triplet = (
            pos_triplets_dataset["head"][i],
            pos_triplets_dataset["relation"][i],
            pos_triplets_dataset["test"][i],
        )
        if (
            valid_triplets is not None and new_triplet in valid_triplets or
            test_triplets is not None and new_triplet in test_triplets or
            new_triplet in pos_triplets
        ):
            i += 1
            continue

        pos_triplets.add(new_triplet)
        i += 1

    neg_triplets = set()
    random.seed(random_seed)
    while len(neg_triplets) < len(pos_triplets):
        head = random.choice(entities)
        tail = random.choice(entities)
        relation = random.choice(relations)
        new_triplet = (head, tail, relation)

        if (
            valid_triplets is not None and new_triplet in valid_triplets or
            test_triplets is not None and new_triplet in test_triplets or
            new_triplet in pos_triplets
        ):
            continue

        neg_triplets.add(new_triplet)

    return {True: pos_triplets, False: neg_triplets}


def _construct_prompts(graph: BaseGraph, prompt_template: str, triplets: dict[bool, set[tuple[str, str, str]]]) -> Dataset:
    d = {"label": [], "text": []}
    for label, label_triplets in triplets.items():
        for t in label_triplets:
            d["label"].append(int(label))
            d["text"] = prompt_template.format(
                graph.entity_id_to_text[t[0]],
                t[1],
                graph.entity_id_to_text[t[1]],
            )

    return Dataset.from_dict(d)



def construct_dataset(
    graph_name: str,
    batch_size: int,
    prompt_template: str,
    cache_dir: str,
    pos_train_size: float = 1.,
    random_seed: int = 42,
    **kwargs,
) -> DatasetDict:
    graph = get_graph(graph_name, batch_size, cache_dir, **kwargs)

    test_triplets = _get_triplets(
        graph.triplets["test"],
        graph.entity_ids,
        graph.relations,
        random_seed=random_seed,
    )
    valid_triplets = _get_triplets(
        graph.triplets["valid"],
        graph.entity_ids,
        graph.relations,
        random_seed=random_seed,
        test_triplets=test_triplets[True] | test_triplets[False],
    )
    train_triplets = _get_triplets(
        graph.triplets["train"],
        entities=graph.entity_ids,
        relations=graph.relations,
        should_shuffle=True,
        size=pos_train_size,
        random_seed=random_seed,
        valid_triplets=valid_triplets[True] | valid_triplets[False],
        test_triplets=test_triplets[True] | test_triplets[False],
    )

    result = DatasetDict()
    result["train"] = _construct_prompts(
        graph,
        prompt_template,
        train_triplets
    )
    result["validation"] = _construct_prompts(
        graph,
        prompt_template,
        valid_triplets
    )
    result["test"] = _construct_prompts(
        graph,
        prompt_template,
        test_triplets
    )

    return result
