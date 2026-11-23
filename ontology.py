from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CATALOG_FILE = DATA_DIR / "ingredient_catalog.csv"
RELATIONS_FILE = DATA_DIR / "ingredient_ontology_relations.csv"
SOURCES_FILE = DATA_DIR / "ontology_sources.csv"
FOOD_INGREDIENT_FILE = DATA_DIR / "food_ingredient_mapping_candidate.csv"

VERIFIED_RELATION_STATUS = {"Verified"}
SEARCHABLE_INGREDIENT_STATUS = {
    "Verified",
    "Verified_Genus_Level_Only",
}


def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    """อ่านข้อมูล CSV โดยรองรับ UTF-8 BOM."""
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล: {file_path}")

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return [
            {
                key: (value or "").strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]


class IngredientOntology:
    """
    ชั้นอ่านข้อมูล Ingredient Ontology จาก CSV

    โมดูลนี้ยังไม่แก้ไขกระบวนการ IR เดิม
    และยังไม่กำหนดว่าจะใช้ Relational DB หรือ RDF/OWL
    """

    def __init__(self) -> None:
        self.catalog = read_csv_rows(CATALOG_FILE)
        self.relations = read_csv_rows(RELATIONS_FILE)
        self.sources = read_csv_rows(SOURCES_FILE)
        self.food_ingredients = read_csv_rows(
            FOOD_INGREDIENT_FILE
        )

        self.catalog_by_id = {
            row["ingredient_id"]: row
            for row in self.catalog
            if row.get("ingredient_id")
        }

        self.sources_by_id = {
            row["source_id"]: row
            for row in self.sources
            if row.get("source_id")
        }

        self._validate_data()

    def _validate_data(self) -> None:
        """ตรวจคอลัมน์หลักและรหัสซ้ำก่อนใช้งาน."""
        required_catalog = {
            "ingredient_id",
            "ingredient_name_as_recorded",
            "canonical_standard_name",
            "verification_status",
        }
        required_relations = {
            "relation_id",
            "subject_id",
            "predicate_relation",
            "object_label",
            "verification_status",
            "evidence_source_id",
        }
        required_food_mapping = {
            "food_id",
            "ingredient_id",
            "ingredient_name",
            "canonical_standard_name",
            "verification_status",
            "mapping_status",
        }

        self._check_columns(
            self.catalog, required_catalog, "Ingredient Catalog"
        )
        self._check_columns(
            self.relations, required_relations, "Ontology Relations"
        )
        self._check_columns(
            self.food_ingredients,
            required_food_mapping,
            "Food Ingredient Mapping",
        )

        self._check_unique(
            self.catalog, "ingredient_id", "Ingredient Catalog"
        )
        self._check_unique(
            self.relations, "relation_id", "Ontology Relations"
        )

    @staticmethod
    def _check_columns(
        rows: list[dict[str, str]],
        required: set[str],
        label: str,
    ) -> None:
        if not rows:
            raise ValueError(f"{label} ไม่มีข้อมูล")

        missing = required - set(rows[0])
        if missing:
            raise ValueError(
                f"{label} ขาดคอลัมน์: {', '.join(sorted(missing))}"
            )

    @staticmethod
    def _check_unique(
        rows: list[dict[str, str]],
        field: str,
        label: str,
    ) -> None:
        values = [
            row.get(field, "")
            for row in rows
        ]
        non_empty = [value for value in values if value]
        if len(non_empty) != len(set(non_empty)):
            raise ValueError(f"{label} มี {field} ซ้ำ")

    def get_verified_relations(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
    ) -> list[dict[str, str]]:
        """คืนเฉพาะความสัมพันธ์ที่ผ่านการตรวจสอบแล้ว."""
        results = []

        for relation in self.relations:
            if relation.get("verification_status") not in VERIFIED_RELATION_STATUS:
                continue

            if subject_id and relation.get("subject_id") != subject_id:
                continue

            if predicate and relation.get("predicate_relation") != predicate:
                continue

            results.append(relation)

        return results

    def get_ingredient(self, ingredient_id: str) -> dict[str, str] | None:
        """ค้นข้อมูลวัตถุดิบด้วยรหัสมาตรฐานใน Catalog."""
        return self.catalog_by_id.get(ingredient_id)

    def get_food_ingredients(
        self,
        food_id: str,
        include_quarantined: bool = False,
    ) -> list[dict[str, Any]]:
        """
        คืนวัตถุดิบที่เชื่อมกับ food_id

        ค่าเริ่มต้นคืนทุกแถวที่ผ่าน IR eligibility gate
        เพื่อไม่ให้ซ่อนเครื่องปรุงหรือส่วนประกอบจริงจากรายละเอียดสูตร
        รายการ Quarantine และ Pending จะไม่ถูกรวม เว้นแต่เรียก
        get_food_mapping_audit() เพื่อการตรวจสอบแยกต่างหาก
        """
        results: list[dict[str, Any]] = []

        for mapping in self.food_ingredients:
            if mapping.get("food_id") != food_id:
                continue

            mapping_status = mapping.get("mapping_status", "")
            allowed_statuses = {
                "ELIGIBLE",
                "ELIGIBLE_WITH_VARIANT_CAVEAT",
                "ELIGIBLE_VECTOR_DUAL_BRIDGE",
            }
            if (
                mapping_status not in allowed_statuses
                and not include_quarantined
            ):
                continue

            ingredient = self.get_ingredient(
                mapping.get("ingredient_id", "")
            ) or {}

            verification_status = mapping.get("verification_status", "")
            results.append({
                **mapping,
                "catalog_record": ingredient,
                "is_searchable_ingredient": (
                    mapping_status == "ELIGIBLE"
                    and verification_status in SEARCHABLE_INGREDIENT_STATUS
                ),
            })

        return results

    def get_food_mapping_audit(
        self,
        food_id: str,
    ) -> list[dict[str, Any]]:
        """คืนทุกแถวของ food_id รวม Quarantine/Pending เพื่อ audit."""
        results: list[dict[str, Any]] = []

        for mapping in self.food_ingredients:
            if mapping.get("food_id") != food_id:
                continue

            ingredient = self.get_ingredient(
                mapping.get("ingredient_id", "")
            ) or {}

            results.append({
                **mapping,
                "catalog_record": ingredient,
            })

        return results

    def get_food_search_terms(self, food_id: str) -> list[str]:
        """
        สร้างคำค้นเสริมจากชื่อวัตถุดิบที่ตรวจสอบแล้ว

        ไม่ใช้ related_plant_parts_and_forms_th เพราะเป็นรูปแบบ
        หรือส่วนพืชที่ไม่ใช่ชื่อพ้อง และไม่ควรถูกยุบเป็นชนิดเดียวกัน
        """
        terms: list[str] = []

        for item in self.get_food_ingredients(food_id):
            if not item.get("is_searchable_ingredient"):
                continue

            catalog = item.get("catalog_record", {})

            candidates = [
                item.get("ingredient_name", ""),
                item.get("canonical_standard_name", ""),
                catalog.get("true_lexical_synonyms_th", ""),
                catalog.get("scientific_name_latin", ""),
                catalog.get("english_name", ""),
            ]

            for candidate in candidates:
                for term in candidate.split("|"):
                    cleaned = term.strip()
                    if cleaned and cleaned not in terms:
                        terms.append(cleaned)

        return terms

    def get_summary(self) -> dict[str, int]:
        """สรุปจำนวนข้อมูลเพื่อใช้ตรวจสอบการโหลด."""
        verified_relations = self.get_verified_relations()

        status_counts: dict[str, int] = {}
        for row in self.food_ingredients:
            status = row.get("mapping_status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "catalog_items": len(self.catalog),
            "relations_total": len(self.relations),
            "relations_verified": len(verified_relations),
            "sources": len(self.sources),
            "food_ingredient_rows": len(self.food_ingredients),
            "eligible_mapping_rows": (
                status_counts.get("ELIGIBLE", 0)
                + status_counts.get("ELIGIBLE_WITH_VARIANT_CAVEAT", 0)
                + status_counts.get("ELIGIBLE_VECTOR_DUAL_BRIDGE", 0)
            ),
            "eligible_exact_rows": status_counts.get("ELIGIBLE", 0),
            "variant_caveat_rows": status_counts.get(
                "ELIGIBLE_WITH_VARIANT_CAVEAT", 0
            ),
            "dual_bridge_rows": status_counts.get(
                "ELIGIBLE_VECTOR_DUAL_BRIDGE", 0
            ),
            "quarantined_mapping_rows": status_counts.get(
                "QUARANTINE_DRIFT_ALERT", 0
            ),
            "pending_mapping_rows": status_counts.get(
                "PENDING_INGREDIENT_VERIFICATION", 0
            ),
        }


if __name__ == "__main__":
    ontology = IngredientOntology()
    print("Ingredient Ontology loaded")
    for key, value in ontology.get_summary().items():
        print(f"{key}: {value}")

    print("\nตัวอย่างวัตถุดิบของเมนู F027")
    for item in ontology.get_food_ingredients("F027"):
        print(
            item["ingredient_name"],
            "->",
            item["canonical_standard_name"],
        )
