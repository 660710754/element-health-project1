from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from source_catalogue import SourceCatalogue

ALLOWED_ROLES = {
    "primary",
    "derived_note",
    "principle",
}

ALLOWED_STATUSES = {
    "verified",
    "review"
}

@dataclass(frozen=True, slots=True)
class FoodEvidence:
    food_id: str
    source_id: str
    page_start: int
    page_end: int
    evidence_role: str
    verification_status: str

class FoodEvidenceCatalogue:
    def __init__(
        self,
        evidence: tuple[FoodEvidence, ...],
    ) -> None:
        self._evidence = evidence

        grouped: dict[str, list[FoodEvidence]] = {}

        for item in evidence:
            grouped.setdefault(item.food_id, []).append(item)

        self._by_food = {
            food_id: tuple(items)
            for food_id, items in grouped.items()
        }

    def __len__(self) -> int:
        return len(self._evidence)

    def for_food(
        self,
        food_id: str,
    ) -> tuple[FoodEvidence, ...]:
        return self._by_food.get(food_id, ())

    @classmethod
    def from_csv(
        cls,
        evidence_path: str | Path,
        *,
        source_catalogue: SourceCatalogue,
    ) -> FoodEvidenceCatalogue:
        records: list[FoodEvidence] = []
        seen_keys: set[tuple[object, ...]] = set()

        with Path(evidence_path).open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {
                "food_id",
                "source_id",
                "page_start",
                "page_end",
                "evidence_role",
                "verification_status",
            }

            missing = required_columns - set(
                reader.fieldnames or ()
            )

            if missing:
                raise ValueError(
                    f"ขาดคอลัมน์: {sorted(missing)}"
                )

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                context = f"บรรทัด {line_number}"

                food_id = _required_text(
                    row.get("food_id"),
                    f"{context}: food_id",
                )
                source_id = _required_text(
                    row.get("source_id"),
                    f"{context}: source_id",
                )
                page_start = _parse_positive_int(
                    row.get("page_start"),
                    f"{context}: page_start",
                )
                page_end = _parse_positive_int(
                    row.get("page_end"),
                    f"{context}: page_end",
                )
                role = _required_text(
                    row.get("evidence_role"),
                    f"{context}: evidence_role",
                )
                status = _required_text(
                    row.get("verification_status"),
                    f"{context}: verification_status",
                )

                source = source_catalogue.require(source_id)

                if page_start > page_end:
                    raise ValueError(
                        f"{context}: page_start มากกว่า page_end"
                    )

                if page_end > source.page_count:
                    raise ValueError(
                        f"{context}: หน้าเกินจำนวนหน้าของ {source_id}"
                    )

                if role not in ALLOWED_ROLES:
                    raise ValueError(
                        f"{context}: evidence_role ไม่ถูกต้อง"
                    )

                if status not in ALLOWED_STATUSES:
                    raise ValueError(
                        f"{context}: verification_status ไม่ถูกต้อง"
                    )

                key = (
                    food_id,
                    source_id,
                    page_start,
                    page_end,
                    role,
                )

                if key in seen_keys:
                    raise ValueError(
                        f"{context}: หลักฐานซ้ำ"
                    )

                seen_keys.add(key)

                records.append(
                    FoodEvidence(
                        food_id=food_id,
                        source_id=source_id,
                        page_start=page_start,
                        page_end=page_end,
                        evidence_role=role,
                        verification_status=status,
                    )
                )

        return cls(tuple(records))
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