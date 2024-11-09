import random
import logging

from datasets import Dataset, DatasetDict

from .base import BaseGraph
from .fb15k_237 import FB15K_237


logger = logging.getLogger(__name__)


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
    max_attempts: int = 10_000_000,
) -> dict[bool, set[tuple[str, str, str]]]:
    pos_triplets_dataset = triplets_dataset
    if should_shuffle:
        pos_triplets_dataset = pos_triplets_dataset.shuffle(seed=random_seed)

    pos_triplets_dataset = pos_triplets_dataset[:int(size * len(pos_triplets_dataset))]

    pos_triplets = set()
    for i in range(len(pos_triplets_dataset["head"])):
        new_triplet = (
            pos_triplets_dataset["head"][i],
            pos_triplets_dataset["relation"][i],
            pos_triplets_dataset["tail"][i],
        )
        if (
            valid_triplets is not None and new_triplet in valid_triplets or
            test_triplets is not None and new_triplet in test_triplets or
            new_triplet in pos_triplets
        ):
            continue

        pos_triplets.add(new_triplet)

    neg_triplets = set()
    random.seed(random_seed)
    n_attempts = 0
    while len(neg_triplets) < len(pos_triplets):
        n_attempts += 1
        if n_attempts > max_attempts:
            logging.info("Reached max attempts to sample negative samples")
            logging.info(f"Generated {len(neg_triplets)} samples out of {len(pos_triplets)}")
            break

        head = random.choice(entities)
        tail = random.choice(entities)
        relation = random.choice(relations)
        new_triplet = (head, relation, tail)

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
            d["text"].append(
                prompt_template.format(
                    graph.entity_id_to_text[t[0]],
                    t[1],
                    graph.entity_id_to_text[t[2]],
                )
            )

    return Dataset.from_dict(d)


def construct_dataset(
    graph_name: str,
    batch_size: int,
    prompt_template: str,
    max_attempts: int,
    cache_dir: str,
    pos_train_size: float = 1.,
    random_seed: int = 42,
    **kwargs,
) -> DatasetDict:
    graph = get_graph(graph_name, batch_size, cache_dir, **kwargs)

    logging.info("Sampling triplets for the test split")
    test_triplets = _get_triplets(
        graph.triplets["test"],
        graph.entity_ids,
        graph.relations,
        random_seed=random_seed,
        max_attempts=max_attempts,
    )

    logging.info("Sampling triplets for the validation split")
    valid_triplets = _get_triplets(
        graph.triplets["validation"],
        graph.entity_ids,
        graph.relations,
        random_seed=random_seed,
        test_triplets=test_triplets[True] | test_triplets[False],
        max_attempts=max_attempts,
    )

    logging.info("Sampling triplets for the training split")
    train_triplets = _get_triplets(
        graph.triplets["train"],
        entities=graph.entity_ids,
        relations=graph.relations,
        should_shuffle=True,
        size=pos_train_size,
        random_seed=random_seed,
        valid_triplets=valid_triplets[True] | valid_triplets[False],
        test_triplets=test_triplets[True] | test_triplets[False],
        max_attempts=max_attempts,
    )

    result = DatasetDict()
    logging.info("Constructing the resulting dataset training split")
    result["train"] = _construct_prompts(
        graph,
        prompt_template,
        train_triplets
    )

    logging.info("Constructing the resulting dataset validation split")
    result["validation"] = _construct_prompts(
        graph,
        prompt_template,
        valid_triplets
    )

    logging.info("Constructing the resulting dataset test split")
    result["test"] = _construct_prompts(
        graph,
        prompt_template,
        test_triplets
    )

    return result
