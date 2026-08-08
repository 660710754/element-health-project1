from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

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


    @classmethod
    def from_csv(
            cls,
            manifest_path: str | Path,
    ) -> SourceCatalogue:
        sources: list[SourceDocument] = []
        seen_ids: set[str] = set()

        with Path(manifest_path).open(
                encoding="utf-8-sig",
                newline="",
        ) as file:
            reader = csv.DictReader(file)

            for line_number, row in enumerate(reader, start=2):
                source_id = _required_text(
                    row.get("source_id"),
                    f"บรรทัด {line_number}: source_id",
                )

                if source_id in seen_ids:
                    raise ValueError(f"source_id ซ้ำ: {source_id}")

                seen_ids.add(source_id)

                sources.append(
                    SourceDocument(
                        source_id=source_id,
                        title_th=_required_text(row.get("title_th"), "title_th"),
                        file_path=_required_text(row.get("file_path"), "file_path"),
                        source_type=_required_text(row.get("source_type"), "source_type"),
                        corpus=_required_text(row.get("corpus"), "corpus"),
                        verification_status=_required_text(
                            row.get("verification_status"),
                            "verification_status",
                        ),
                        index_enabled=_parse_bool(
                            _required_text(row.get("index_enabled"), "index_enabled"),
                            "index_enabled",
                        ),
                        page_count=_parse_positive_int(
                            row.get("page_count"),
                            "page_count",
                        ),
                        sha256=_required_text(row.get("sha256"), "sha256"),
                        extraction_status=_required_text(
                            row.get("extraction_status"),
                            "extraction_status",
                        ),
                        notes=(row.get("notes") or "").strip() or None,
                    )
                )

        return cls(tuple(sources))


def _parse_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()

    if normalized not in {"true", "false"}:
        raise ValueError(
            f"{context} ต้องเป็น true หรือ false"
        )

    return normalized == "true"

def _required_text(
    value: str | None,
    context: str,
) -> str:
    text = (value or "").strip()

    if not text:
        raise ValueError(f"{context} ห้ามเป็นค่าว่าง")

    return text


def _parse_positive_int(
    value: str | None,
    context: str,
) -> int:
    text = _required_text(value, context)

    try:
        number = int(text)
    except ValueError as error:
        raise ValueError(
            f"{context} ต้องเป็นจำนวนเต็ม"
        ) from error

    if number <= 0:
        raise ValueError(
            f"{context} ต้องมากกว่า 0"
        )

    return number