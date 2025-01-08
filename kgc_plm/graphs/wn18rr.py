import logging

from datasets import Dataset, DatasetDict

from .base import BaseGraph

logger = logging.getLogger(__name__)


class WN18RR(BaseGraph):
    def __init__(
        self,
        data_path: str = "data/wn18rr",
        add_reverse_relations: bool = False,
        cache_dir: str = "cache",
        **kwargs,
    ) -> None:
        logging.info("Loading the dataset triplets")
        self._load_triplets_dataset(
            data_path=data_path,
            add_reverse_relations=add_reverse_relations,
            cache_dir=cache_dir
        )
        logging.info("Finished loading the dataset!")

    @property
    def triplets(self) -> DatasetDict:
        return self._triplets_dataset

    @property
    def entity_id_to_text(self) -> dict[str, str]:
        return self._entityid_to_name

    @property
    def entity_ids(self) -> list[str]:
        return self._entity_names

    @property
    def texts(self) -> list[str]:
        return self._entity_names

    @property
    def relations(self) -> list[str]:
        return self._relations

    def _load_triplets_dataset(
            self, data_path: str, add_reverse_relations: bool, cache_dir: str
        ) -> None:
        entity_names = set()
        relation_names = set()

        dataset = {}
        for split_filename in ["train", "valid", "test"]:
            split = {
                "head": [],
                "relation": [],
                "tail": [],
            }
            with open(data_path + "/" + split_filename + ".txt") as file:
                for line in file:
                    head, relation, tail = line.strip().split("\t")
                    if head[0] == "'":
                        head = head[1:]
                    if tail[0] == "'":
                        tail = tail[1:]
                    head = head.strip()
                    relation = relation.strip()
                    tail = tail.strip()

                    entity_names.add(head)
                    entity_names.add(tail)
                    relation_names.add(relation)

                    split["head"].append(head)
                    split["relation"].append(relation)
                    split["tail"].append(tail)

                    if add_reverse_relations:
                        split["head"].append(head)
                        split["relation"].append(f"{relation}_reverse")
                        split["tail"].append(tail)

            split = Dataset.from_dict(split, cache_dir=cache_dir)
            dataset[split_filename if split_filename != "valid" else "validation"] = split

        self._entity_names = sorted(entity_names)
        self._relations = sorted(relation_names)
        self._triplets_dataset = DatasetDict(dataset)
        self._entityid_to_name = {entity_name: entity_name for entity_name in self._entity_names}
