from datasets import DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

from .base import BaseGraph


class FB15K_237(BaseGraph):
    def __init__(
        self,
        batch_size: int = 1000,
        add_reverse_relations: bool = False,
        cache_dir: str = "cache",
    ) -> None:
        self._triplets_dataset = FB15K_237._load_triplets_dataset(
            batch_size=batch_size,
            add_reverse_relations=add_reverse_relations,
            cache_dir=cache_dir,
        )

        self._entityid_to_description = FB15K_237._load_mid2description(cache_dir)

        # some entities do not have descriptions, so we use names for them
        self._entityid_to_name = FB15K_237._load_mid2name(cache_dir)
        for entity_id, name in self._entityid_to_name.items():
            if entity_id not in self._entityid_to_description:
                self._entityid_to_description[entity_id] = name

        self._entityids = []
        self._descriptions = []
        for entity_id, description in sorted(self._entityid_to_description.items()):
            self._entityids.append(entity_id)
            self._descriptions.append(description)

        all_relations = set()
        for split_name in self._triplets_dataset:
            all_relations |= set(self._triplets_dataset[split_name]["relation"])

        self._relations = sorted(all_relations)

    @property
    def triplets(self) -> DatasetDict:
        return self._triplets_dataset

    @property
    def entity_id_to_text(self) -> dict[str, str]:
        # Descriptions seem to already contain the entity names
        # in some shape or form, so we can try using only them.
        return self._entityid_to_description

    @property
    def entity_ids(self) -> list[str]:
        return self._entityids

    @property
    def texts(self) -> list[str]:
        return self._descriptions

    @property
    def relations(self) -> list[str]:
        return self._relations

    @staticmethod
    def _load_triplets_dataset(
        batch_size: int, add_reverse_relations: bool, cache_dir: str
    ) -> DatasetDict:
        triplets_dataset = load_dataset("KGraph/FB15k-237", cache_dir=cache_dir)
        result = triplets_dataset.map(
            lambda items: FB15K_237._transform_triplets_dataset(
                items=items, invert_triplets=False
            ),
            batched=True,
            batch_size=batch_size,
        ).remove_columns("text")
        if add_reverse_relations:
            inverted_triplets_dataset = triplets_dataset.map(
                lambda items: FB15K_237._transform_triplets_dataset(
                    items=items, invert_triplets=True
                ),
                batched=True,
                batch_size=batch_size,
            ).remove_columns("text")
            result = DatasetDict(
                {
                    split_name: concatenate_datasets(
                        [
                            result[split_name],
                            inverted_triplets_dataset[split_name],
                        ]
                    )
                    for split_name in result
                }
            )

        return result

    @staticmethod
    def _transform_triplets_dataset(
        items: dict[str, list[str]], invert_triplets: bool
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {
            "head": [],
            "relation": [],
            "tail": [],
        }
        for item in items["text"]:
            head_id, relation, tail_id = item.split("\t")
            if invert_triplets:
                result["head"].append(tail_id)
                result["relation"].append(f"{relation}_reverse")
                result["tail"].append(head_id)
            else:
                result["head"].append(head_id)
                result["relation"].append(relation)
                result["tail"].append(tail_id)

        return result

    @staticmethod
    def _load_mid2name(cache_dir: str) -> dict[str, str]:
        mid2name_path = hf_hub_download(
            repo_id="KGraph/FB15k-237",
            filename="data/FB15k_mid2name.txt",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        mid2name = {}
        with open(mid2name_path) as file:
            for line in file:
                entity_id, entity_name = line.strip().split("\t")
                mid2name[entity_id] = entity_name

        return mid2name

    @staticmethod
    def _load_mid2description(cache_dir: str) -> dict[str, str]:
        mid2description_path = hf_hub_download(
            repo_id="KGraph/FB15k-237",
            filename="data/FB15k_mid2description.txt",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        mid2description = {}
        with open(mid2description_path) as file:
            for line in file:
                entity_id, entity_description = line.strip().split("\t")
                entity_description = entity_description[1 : -len('"@en')]
                mid2description[entity_id] = entity_description

        return mid2description
