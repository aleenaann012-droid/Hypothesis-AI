from abc import ABC, abstractmethod


class RelationProvider(ABC):
    """Common interface for local and hosted extraction models."""

    @abstractmethod
    def extract(self, sentence: str) -> list[dict]:
        """Return candidate subject-predicate-object triples."""
        raise NotImplementedError