from abc import ABC, abstractmethod

from datasets import DatasetDict


class BaseGraph(ABC):
    @property
    @abstractmethod
    def relations(self) -> DatasetDict:
        ...

    @property
    @abstractmethod
    def entity_id_to_text(self) -> dict[str, str]:
        ...

    @property
    @abstractmethod
    def entity_ids(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def texts(self) -> list[str]:
        ...