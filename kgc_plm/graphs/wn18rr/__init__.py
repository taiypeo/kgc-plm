import logging

from datasets import Dataset, DatasetDict

from ..base import BaseGraph

logger = logging.getLogger(__name__)


class WN18RR(BaseGraph):
    def __init__(
        self,
        add_reverse_relations: bool = False,
        cache_dir: str = "cache",
    ) -> None:
        logging.info("Loading the dataset triplets")
        self._load_triplets_dataset(add_reverse_relations=add_reverse_relations, cache_dir=cache_dir)
        logging.info("Finished loading the dataset!")

    @property
    def triplets(self) -> DatasetDict:
        return self._triplets_dataset

    @property
    def entity_id_to_text(self) -> dict[str, str]:
        return self._entityid_to_name

    @property
    def entity_ids(self) -> list[str]:
        return self._entityids

    @property
    def texts(self) -> list[str]:
        return self._entity_names

    @property
    def relations(self) -> list[str]:
        return self._relations

    def _load_triplets_dataset(self, add_reverse_relations: bool, cache_dir: str) -> None:
        entity_names = set()
        relation_names = set()

        dataset = {}
        for split_filename in ["train", "valid", "test"]:
            with open("data/" + split_filename + ".txt") as file:
                def _generate_split():
                    for line in file:
                        head, relation, tail = line.strip().split("\t")
                        entity_names.add(head)
                        entity_names.add(tail)
                        relation_names.add(relation)

                        yield {
                            "head": head,
                            "relation": relation,
                            "tail": tail,
                        }

                        if add_reverse_relations:
                            yield {
                                "head": tail,
                                "relation": relation,
                                "tail": head,
                            }

                split = Dataset.from_generator(_generate_split, cache_dir=cache_dir)
                dataset[split_filename if split_filename != "valid" else "validation"] = split

        self._entity_names = sorted(entity_names)
        self._relations = sorted(relation_names)
        self._triplets_dataset = DatasetDict(dataset)
        self._entityids = [str(i) for i in range(len(self._entity_names))]
        self._entityid_to_name = {str(i): entity_name for i, entity_name in enumerate(self._entity_names)}
