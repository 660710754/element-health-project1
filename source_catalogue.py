from __future__ import annotations

# import csv
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SourceDocument:
    source_id: str
    title_th: str
    file_path: str
    source_type: str
    corpus: str
    verification_status: str
    index_enabled: bool
    page_count: int
    sha256: str
    extraction_status: str
    notes: str | None

    @property
    def is_indexable(self) -> bool:
        return self.index_enabled

class SourceCatalogue:
    def __init__(
        self,
        sources: tuple[SourceDocument, ...],
    ) -> None:
        self._sources = sources
        self._by_id = {
            source.source_id: source
            for source in sources
        }

    def __len__(self) -> int:
        return len(self._sources)

    def get(
        self,
        source_id: str,
    ) -> SourceDocument | None:
        return self._by_id.get(source_id)

    def require(
        self,
        source_id: str,
    ) -> SourceDocument:
        source = self.get(source_id)

        if source is None:
            raise KeyError(f"ไม่พบ source_id: {source_id}")

        return source

    def indexable_sources(
        self,
    ) -> tuple[SourceDocument, ...]:
        return tuple(
            source
            for source in self._sources
            if source.is_indexable
        )

def _parse_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()

    if normalized not in {"true", "false"}:
        raise ValueError(
            f"{context} ต้องเป็น true หรือ false"
        )

    return normalized == "true"