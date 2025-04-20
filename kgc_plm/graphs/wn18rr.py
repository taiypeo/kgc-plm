import logging
import re

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi

from .base import BaseGraph

logger = logging.getLogger(__name__)
pattern = re.compile(r"[\W_]+")
split_pattern = re.compile(r"[-_\s]+")


def tokenizer(doc: str) -> list[str]:
    return [pattern.sub("", token.lower()) for token in split_pattern.split(doc)]


class WN18RR(BaseGraph):
    def __init__(
        self,
        data_path: str = "data/wn18rr",
        add_reverse_relations: bool = False,
        cache_dir: str = "cache",
        use_freebase_descriptions_as_texts: bool = False,
        use_wordnet_descriptions_as_texts: bool = False,
        **kwargs,
    ) -> None:
        self._use_freebase_descriptions_as_texts = use_freebase_descriptions_as_texts
        self._use_wordnet_descriptions_as_texts = use_wordnet_descriptions_as_texts

        if self._use_freebase_descriptions_as_texts:
            logging.info("Loading the Freebase descriptions")
            self._descriptions = WN18RR._load_freebase_descriptions(cache_dir)
            self._bm25 = BM25Okapi(self._descriptions, tokenizer=tokenizer)
        elif self._use_wordnet_descriptions_as_texts:
            logging.info("Loading the WordNet descriptions")
            self._descriptions = WN18RR._load_wordnet_descriptions(cache_dir)
            self._bm25 = BM25Okapi(self._descriptions, tokenizer=tokenizer)

        logging.info("Loading the dataset triplets")
        self._load_triplets_dataset(
            data_path=data_path,
            add_reverse_relations=add_reverse_relations,
        )
        logging.info("Finished loading the dataset!")

    @property
    def triplets(self) -> DatasetDict:
        return self._triplets_dataset

    @property
    def entity_id_to_text(self) -> dict[str, str]:
        return self._entity_name_to_description

    @property
    def entity_ids(self) -> list[str]:
        return self._entity_names

    @property
    def texts(self) -> list[str]:
        return self._descriptions

    @property
    def relations(self) -> list[str]:
        return self._relations

    def _load_triplets_dataset(
            self, data_path: str, add_reverse_relations: bool
        ) -> None:
        self._entity_name_to_description = {}
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

                    relation_names.add(relation)
                    if self._use_freebase_descriptions_as_texts or self._use_wordnet_descriptions_as_texts:
                        head_query = tokenizer(head.split(".")[0])
                        tail_query = tokenizer(tail.split(".")[0])
                        self._entity_name_to_description[head] = self._bm25.get_top_n(head_query, self._descriptions, n=1)[0]
                        self._entity_name_to_description[tail] = self._bm25.get_top_n(tail_query, self._descriptions, n=1)[0]
                    else:
                        self._entity_name_to_description[head] = head
                        self._entity_name_to_description[tail] = tail

                    split["head"].append(head)
                    split["relation"].append(relation)
                    split["tail"].append(tail)

                    if add_reverse_relations:
                        split["head"].append(head)
                        split["relation"].append(f"{relation}_reverse")
                        split["tail"].append(tail)

            split = Dataset.from_dict(split)
            dataset[split_filename if split_filename != "valid" else "validation"] = split

        self._entity_names = []
        self._descriptions = []
        for entity_name, description in sorted(self._entity_name_to_description.items()):
            self._entity_names.append(entity_name)
            self._descriptions.append(description)

        self._relations = sorted(relation_names)
        self._triplets_dataset = DatasetDict(dataset)

    @staticmethod
    def _load_freebase_descriptions(cache_dir: str) -> list[str]:
        mid2description_path = hf_hub_download(
            repo_id="KGraph/FB15k-237",
            filename="data/FB15k_mid2description.txt",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        descriptions = []
        with open(mid2description_path) as file:
            for line in file:
                _, entity_description = line.strip().split("\t")
                entity_description = entity_description[1 : -len('"@en')]
                descriptions.append(entity_description)

        return descriptions

    @staticmethod
    def _load_wordnet_descriptions(cache_dir: str) -> list[str]:
        def gen_text(example):
            example["text"] = (example["Word"] or "") + " " + (example["Definition"] or "")
            return example

        dataset = load_dataset("marksverdhei/wordnet-definitions-en-2021", cache_dir=cache_dir)
        dataset_transformed = dataset.map(gen_text)

        return dataset_transformed["train"]["text"] + dataset_transformed["validation"]["text"] + dataset_transformed["test"]["text"]
