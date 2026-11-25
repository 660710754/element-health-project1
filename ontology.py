from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CATALOG_FILE = DATA_DIR / "ingredient_catalog.csv"
RELATIONS_FILE = DATA_DIR / "ingredient_ontology_relations.csv"
SOURCES_FILE = DATA_DIR / "ontology_sources.csv"

RECIPES_FILE = DATA_DIR / "recipes_3nf.csv"
RECIPE_INGREDIENTS_FILE = DATA_DIR / "recipe_ingredients_3nf.csv"
SEARCH_TERMS_FILE = DATA_DIR / "ingredient_search_terms.csv"

VERIFIED_RELATION_STATUS = {"Verified"}

SEARCHABLE_INGREDIENT_STATUS = {
    "Verified",
    "Verified_Genus_Level_Only",
}

ALLOWED_RECIPE_STATUSES = {
    "ELIGIBLE",
    "ELIGIBLE_WITH_VARIANT_CAVEAT",
    "ELIGIBLE_VECTOR_DUAL_BRIDGE",
    "QUARANTINE_DRIFT_ALERT",
}


# =========================================================
# CSV Reader
# =========================================================

def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    """อ่านข้อมูล CSV โดยรองรับ UTF-8 BOM."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ข้อมูล: {file_path}"
        )

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


# =========================================================
# Ingredient Ontology
# =========================================================

class IngredientOntology:
    """
    อ่านข้อมูล Ingredient Ontology จากโครงสร้าง 3NF

    Tables:
        1. Ingredient Catalog
        2. Ontology Relations
        3. Ontology Sources
        4. Recipes
        5. Recipe Ingredients

    IR eligibility เป็น Derived Data
    ไม่จัดเก็บซ้ำในตาราง Recipe Ingredients
    """

    def __init__(self) -> None:

        self.catalog = read_csv_rows(CATALOG_FILE)
        self.relations = read_csv_rows(RELATIONS_FILE)
        self.sources = read_csv_rows(SOURCES_FILE)

        self.recipes = read_csv_rows(RECIPES_FILE)

        self.recipe_ingredients = read_csv_rows(
            RECIPE_INGREDIENTS_FILE
        )

        # Search Expansion จาก Canonical Ontology Dataset
        self.search_terms = read_csv_rows(SEARCH_TERMS_FILE)

        self.search_terms_by_ingredient: dict[str, list[str]] = {}

        for row in self.search_terms:
            ingredient_id = row["ingredient_id"]
            term = row["search_term"]

            self.search_terms_by_ingredient.setdefault(
                ingredient_id, []
            )

            if term and term not in self.search_terms_by_ingredient[ingredient_id]:
                self.search_terms_by_ingredient[ingredient_id].append(term)

        # Ingredient Catalog lookup
        self.catalog_by_id = {
            row["ingredient_id"]: row
            for row in self.catalog
            if row.get("ingredient_id")
        }

        # Recipe lookup
        self.recipes_by_id = {
            row["recipe_id"]: row
            for row in self.recipes
            if row.get("recipe_id")
        }

        # Source lookup
        self.sources_by_id = {
            row["source_id"]: row
            for row in self.sources
            if row.get("source_id")
        }

        self._validate_data()

        # สร้างข้อมูลจากการ JOIN ตาราง 3NF
        self._build_food_ingredients()

    # =====================================================
    # Data Validation
    # =====================================================

    def _validate_data(self) -> None:
        """ตรวจสอบคอลัมน์ Primary Key และ Foreign Key."""

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

        required_sources = {
            "source_id",
            "title_th",
            "verification_status",
        }

        required_recipes = {
            "recipe_id",
            "food_id",
            "food_name",
            "recipe_audit_status",
            "is_drift_alert",
            "recipe_ir_status",
        }

        required_recipe_ingredients = {
            "recipe_id",
            "ingredient_id",
            "ingredient_name_as_recorded",
            "quantity",
            "unit",
            "ingredient_role",
            "option_group_normalized",
        }

        required_search_terms = {
            "ingredient_id",
            "search_term",
        }

        self._check_columns(
            self.catalog,
            required_catalog,
            "Ingredient Catalog",
        )

        self._check_columns(
            self.relations,
            required_relations,
            "Ontology Relations",
        )

        self._check_columns(
            self.sources,
            required_sources,
            "Ontology Sources",
        )

        self._check_columns(
            self.recipes,
            required_recipes,
            "Recipes",
        )

        self._check_columns(
            self.recipe_ingredients,
            required_recipe_ingredients,
            "Recipe Ingredients",
        )

        self._check_columns(
            self.search_terms,
            required_search_terms,
            "Ingredient Search Terms",
        )

        # ตรวจสอบ Composite Key ของ Search Terms
        search_term_keys = [
            (row["ingredient_id"], row["search_term"])
            for row in self.search_terms
        ]

        if len(search_term_keys) != len(set(search_term_keys)):
            raise ValueError(
                "Ingredient Search Terms มีคู่ ingredient_id/search_term ซ้ำ"
            )

        # ตรวจสอบ Foreign Key ไปยัง Ingredient Catalog
        for row in self.search_terms:
            if row["ingredient_id"] not in self.catalog_by_id:
                raise ValueError(
                    "ไม่พบ ingredient_id ใน Search Terms: "
                    f"{row['ingredient_id']}"
                )

        # Primary Key
        self._check_unique(
            self.catalog,
            "ingredient_id",
            "Ingredient Catalog",
        )

        self._check_unique(
            self.relations,
            "relation_id",
            "Ontology Relations",
        )

        self._check_unique(
            self.sources,
            "source_id",
            "Ontology Sources",
        )

        self._check_unique(
            self.recipes,
            "recipe_id",
            "Recipes",
        )

        # Composite Primary Key
        mapping_keys = [
            (
                row["recipe_id"],
                row["ingredient_id"],
            )
            for row in self.recipe_ingredients
        ]

        if len(mapping_keys) != len(set(mapping_keys)):
            raise ValueError(
                "Recipe Ingredients มี Composite Key ซ้ำ"
            )

        # Foreign Key: Recipe
        for row in self.recipe_ingredients:

            if row["recipe_id"] not in self.recipes_by_id:
                raise ValueError(
                    f"ไม่พบ recipe_id: {row['recipe_id']}"
                )

        # Foreign Key: Ingredient
        for row in self.recipe_ingredients:

            if row["ingredient_id"] not in self.catalog_by_id:
                raise ValueError(
                    f"ไม่พบ ingredient_id: {row['ingredient_id']}"
                )

        # Recipe status
        for recipe in self.recipes:

            if recipe["recipe_ir_status"] not in ALLOWED_RECIPE_STATUSES:
                raise ValueError(
                    "สถานะ Recipe ไม่ถูกต้อง: "
                    f"{recipe['recipe_id']} "
                    f"{recipe['recipe_ir_status']}"
                )

    @staticmethod
    def _check_columns(
        rows: list[dict[str, str]],
        required: set[str],
        label: str,
    ) -> None:
        """ตรวจสอบคอลัมน์ที่จำเป็น."""

        if not rows:
            raise ValueError(
                f"{label} ไม่มีข้อมูล"
            )

        missing = required - set(rows[0])

        if missing:
            raise ValueError(
                f"{label} ขาดคอลัมน์: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _check_unique(
        rows: list[dict[str, str]],
        field: str,
        label: str,
    ) -> None:
        """ตรวจสอบ Primary Key."""

        values = [
            row.get(field, "")
            for row in rows
        ]

        if not all(values):
            raise ValueError(
                f"{label} มีค่า {field} ว่าง"
            )

        if len(values) != len(set(values)):
            raise ValueError(
                f"{label} มี {field} ซ้ำ"
            )

    # =====================================================
    # Build Derived Food Ingredients
    # =====================================================

    def _build_food_ingredients(self) -> None:
        """
        JOIN Recipes + Recipe Ingredients + Catalog

        คำนวณ IR eligibility จากสถานะของตารางแม่
        โดยไม่เก็บสถานะซ้ำในตารางเชื่อม
        """

        self.food_ingredients = []

        for mapping in self.recipe_ingredients:

            recipe_id = mapping["recipe_id"]
            ingredient_id = mapping["ingredient_id"]

            recipe = self.recipes_by_id[recipe_id]
            ingredient = self.catalog_by_id[ingredient_id]

            recipe_status = recipe["recipe_ir_status"]

            verification_status = ingredient[
                "verification_status"
            ]

            # Derived IR eligibility
            if verification_status == "Pending_Verification":

                ir_status = "PENDING_INGREDIENT_VERIFICATION"

            else:

                ir_status = recipe_status

            row = {
                **mapping,

                "food_id": recipe["food_id"],
                "food_name": recipe["food_name"],

                "ingredient_name": mapping[
                    "ingredient_name_as_recorded"
                ],

                "canonical_standard_name": ingredient[
                    "canonical_standard_name"
                ],

                "verification_status": verification_status,

                "mapping_status": ir_status,

                "recipe_audit_status": recipe[
                    "recipe_audit_status"
                ],

                "is_drift_alert": recipe[
                    "is_drift_alert"
                ],

                "catalog_record": ingredient,
                "recipe_record": recipe,
            }

            self.food_ingredients.append(row)

    # =====================================================
    # Ontology Relations
    # =====================================================

    def get_verified_relations(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
    ) -> list[dict[str, str]]:
        """คืนเฉพาะความสัมพันธ์ที่ผ่านการตรวจสอบแล้ว."""

        results = []

        for relation in self.relations:

            if (
                relation.get("verification_status")
                not in VERIFIED_RELATION_STATUS
            ):
                continue

            if (
                subject_id
                and relation.get("subject_id") != subject_id
            ):
                continue

            if (
                predicate
                and relation.get("predicate_relation") != predicate
            ):
                continue

            results.append(relation)

        return results


    # =====================================================
    # Ontology Inference Engine
    # =====================================================

    def infer_ingredient_facts(
        self,
        ingredient_id: str,
    ) -> list[dict[str, Any]]:
        """
        อนุมานข้อเท็จจริงของวัตถุดิบจาก Verified Relations

        รองรับ:
        - Same as
        - Synonym of
        - Has scientific name
        - Associated with element
        - Part of
        - Derived from
        - Decomposed into
        - Variant of

        ไม่อนุมานจาก Pending หรือ Unverified
        และไม่เปลี่ยนความสัมพันธ์ให้เป็น Same as
        """

        ingredient = self.get_ingredient(ingredient_id)

        if not ingredient:
            return []

        canonical_name = ingredient.get(
            "canonical_standard_name", ""
        )

        # ค้นหาความสัมพันธ์ของชื่อที่บันทึก
        # และชื่อมาตรฐานของวัตถุดิบ
        relations = [
            row
            for row in self.relations
            if (
                row.get("subject_id") == ingredient_id
                or (
                    canonical_name
                    and row.get("subject_id") == canonical_name
                )
            )
            and row.get("verification_status") == "Verified"
        ]

        results = []

        for relation in relations:

            predicate = relation["predicate_relation"]

            # ใช้เฉพาะ Relationship ที่รองรับ
            supported_predicates = {
                "Same as",
                "Synonym of",
                "Has scientific name",
                "Associated with element",
                "Part of",
                "Derived from",
                "Decomposed into",
                "Variant of",
            }

            if predicate not in supported_predicates:
                continue

            results.append({
                "ingredient_id": ingredient_id,
                "ingredient_name": ingredient[
                    "ingredient_name_as_recorded"
                ],
                "canonical_name": canonical_name,
                "relationship": predicate,
                "related_entity": relation["object_label"],
                "related_entity_type": relation[
                    "object_entity_type"
                ],
                "relation_id": relation["relation_id"],
                "evidence_source_id": relation[
                    "evidence_source_id"
                ],
                "inference_status": "VERIFIED",
            })

        return results


    # =====================================================
    # Inference Audit
    # =====================================================

    def get_inference_audit(
        self,
        ingredient_id: str,
    ) -> list[dict[str, Any]]:
        """
        ตรวจสอบย้อนกลับผลการอนุมาน

        แสดง:
        - Ingredient
        - Relationship ID
        - Relationship Type
        - Inferred Result
        - Evidence Source
        - Evidence Page
        - Evidence Note

        แยก Source ID เมื่อมีหลายแหล่ง
        และไม่รวม Pending หรือ Unverified Relations
        """

        facts = self.infer_ingredient_facts(
            ingredient_id
        )

        if not facts:
            return []

        # สร้าง Source Lookup
        source_by_id = {
            row["source_id"]: row
            for row in self.sources
            if row.get("source_id")
        }

        # สร้าง Relationship Lookup
        relation_by_id = {
            row["relation_id"]: row
            for row in self.relations
            if row.get("relation_id")
        }

        audit_results = []

        for fact in facts:

            relation = relation_by_id.get(
                fact["relation_id"]
            )

            if not relation:
                continue

            if relation.get(
                "verification_status"
            ) != "Verified":
                continue

            raw_source_ids = relation.get(
                "evidence_source_id", ""
            )

            source_ids = [
                sid.strip()
                for sid in raw_source_ids.split("|")
                if sid.strip()
            ]

            if not source_ids:
                source_ids = ["NONE"]

            for source_id in source_ids:

                source = source_by_id.get(
                    source_id
                )

                if source_id == "NONE":
                    audit_status = "NO_SOURCE_RECORDED"
                elif source:
                    audit_status = "SOURCE_FOUND"
                else:
                    audit_status = "SOURCE_NOT_FOUND"

                audit_results.append({

                    "ingredient_id": ingredient_id,

                    "ingredient_name": fact[
                        "ingredient_name"
                    ],

                    "canonical_name": fact[
                        "canonical_name"
                    ],

                    "relation_id": fact[
                        "relation_id"
                    ],

                    "relationship": fact[
                        "relationship"
                    ],

                    "inferred_result": fact[
                        "related_entity"
                    ],

                    "related_entity_type": fact[
                        "related_entity_type"
                    ],

                    "evidence_source_id": source_id,

                    "source_title": (
                        source.get("title_th", "")
                        if source else ""
                    ),

                    "source_type": (
                        source.get("source_type", "")
                        if source else ""
                    ),

                    "source_location": (
                        source.get(
                            "access_url_or_holding", ""
                        )
                        if source else ""
                    ),

                    "evidence_page_or_section": relation.get(
                        "evidence_page_or_section", ""
                    ),

                    "evidence_note": relation.get(
                        "evidence_note", ""
                    ),

                    "verification_status": relation[
                        "verification_status"
                    ],

                    "audit_status": audit_status,

                })

        return audit_results

    # =====================================================
    # Element Inference
    # =====================================================

    def infer_ingredient_elements(
        self,
        ingredient_id: str,
    ) -> list[str]:
        """
        อนุมานธาตุจาก Associated with element

        คืนเฉพาะธาตุที่มี Verified Relation
        """

        facts = self.infer_ingredient_facts(
            ingredient_id
        )

        elements = []

        for fact in facts:

            if fact["relationship"] != "Associated with element":
                continue

            element = fact["related_entity"]

            if element not in elements:
                elements.append(element)

        return elements

    # =====================================================
    # Ingredient Lookup
    # =====================================================

    def get_ingredient(
        self,
        ingredient_id: str,
    ) -> dict[str, str] | None:
        """ค้นหาวัตถุดิบด้วย ingredient_id."""

        return self.catalog_by_id.get(ingredient_id)

    # =====================================================
    # Food Ingredients
    # =====================================================

    def get_food_ingredients(
        self,
        food_id: str,
        include_quarantined: bool = False,
    ) -> list[dict[str, Any]]:
        """
        คืนวัตถุดิบของเมนู

        ค่าเริ่มต้น:
            ELIGIBLE
            ELIGIBLE_WITH_VARIANT_CAVEAT
            ELIGIBLE_VECTOR_DUAL_BRIDGE

        include_quarantined=True:
            คืนทุกสถานะเพื่อการตรวจสอบ
        """

        results = []

        allowed_statuses = {
            "ELIGIBLE",
            "ELIGIBLE_WITH_VARIANT_CAVEAT",
            "ELIGIBLE_VECTOR_DUAL_BRIDGE",
        }

        for mapping in self.food_ingredients:

            if mapping["food_id"] != food_id:
                continue

            mapping_status = mapping["mapping_status"]

            if (
                mapping_status not in allowed_statuses
                and not include_quarantined
            ):
                continue

            ingredient = mapping["catalog_record"]

            verification_status = mapping[
                "verification_status"
            ]

            results.append({
                **mapping,

                "catalog_record": ingredient,

                "is_searchable_ingredient": (
                    mapping_status == "ELIGIBLE"
                    and verification_status
                    in SEARCHABLE_INGREDIENT_STATUS
                ),
            })

        return results

    # =====================================================
    # Food Mapping Audit
    # =====================================================

    def get_food_mapping_audit(
        self,
        food_id: str,
    ) -> list[dict[str, Any]]:
        """คืนข้อมูลทุกสถานะของเมนูเพื่อการตรวจสอบ."""

        return [
            {
                **mapping,
                "catalog_record": mapping["catalog_record"],
            }
            for mapping in self.food_ingredients
            if mapping["food_id"] == food_id
        ]

    # =====================================================
    # Search Term Expansion
    # =====================================================

    def _add_term(
        self,
        terms: list[str],
        candidate: str,
    ) -> None:
        """เพิ่มคำค้นโดยไม่ให้ซ้ำ."""

        for term in candidate.split("|"):

            cleaned = term.strip()

            if cleaned and cleaned not in terms:
                terms.append(cleaned)

    def _is_searchable_component(
        self,
        component_name: str,
    ) -> bool:
        """ตรวจสอบว่าส่วนประกอบมีข้อมูล Catalog ที่ค้นหาได้."""

        return any(
            row["canonical_standard_name"] == component_name
            and row["verification_status"]
            in SEARCHABLE_INGREDIENT_STATUS
            for row in self.catalog
        )

    def _add_part_aware_terms(
        self,
        terms: list[str],
        catalog: dict[str, str],
    ) -> None:
        """
        เพิ่มคำค้นตามส่วนของพืชที่ระบุว่าใช้จริง

        ไม่ถือว่าชื่อส่วนพืชเป็น lexical synonym
        และไม่สร้างความสัมพันธ์ Same as เพิ่มเอง
        """

        plant_part = catalog.get(
            "plant_part_used_th", ""
        ).strip().lower()

        related_parts = catalog.get(
            "related_plant_parts_and_forms_th", ""
        )

        if not plant_part or not related_parts:
            return

        part_types = (
            "ใบ",
            "ดอก",
            "ต้น",
            "ยอด",
            "ลูก",
            "ราก",
            "เหง้า",
        )

        for candidate in related_parts.split("|"):
            term = candidate.strip()

            if not term:
                continue

            # เพิ่มชื่อรวมเมื่อมีการใช้ทั้งใบและดอก
            if term.startswith("ใบและดอก"):
                if "ใบ" in plant_part and "ดอก" in plant_part:
                    self._add_term(terms, term)
                continue

            # เพิ่มเฉพาะส่วนพืชที่สอดคล้องกับข้อมูลการใช้จริง
            matched_part = next(
                (
                    part
                    for part in part_types
                    if term.startswith(part)
                ),
                None,
            )

            if matched_part and matched_part in plant_part:
                self._add_term(terms, term)

    def get_food_search_terms(
        self,
        food_id: str,
    ) -> list[str]:
        """
        สร้างคำค้นเสริมจาก Ingredient Ontology และ Search Expansion

        รองรับ:
            - ชื่อวัตถุดิบและ Canonical name
            - ชื่อพ้อง ชื่อวิทยาศาสตร์ และชื่อภาษาอังกฤษ
            - Search Expansion จาก ingredient_search_terms.csv
            - Part-aware Search Expansion
            - Verified Same as และ Decomposed into

        ใช้เฉพาะ mapping status ที่อนุญาตให้ค้นหา
        และใช้เฉพาะ Ontology Relations ที่ผ่าน Verified
        """

        terms: list[str] = []

        allowed_mapping_statuses = {
            "ELIGIBLE",
            "ELIGIBLE_WITH_VARIANT_CAVEAT",
            "ELIGIBLE_VECTOR_DUAL_BRIDGE",
        }

        for item in self.get_food_ingredients(food_id):

            mapping_status = item["mapping_status"]
            verification_status = item["verification_status"]
            ingredient_id = item["ingredient_id"]

            # ข้าม Quarantine และ Pending
            if mapping_status not in allowed_mapping_statuses:
                continue

            catalog = item["catalog_record"]

            # ---------------------------------------------
            # Search Expansion จาก Canonical Ontology Dataset
            # ใช้เฉพาะวัตถุดิบที่ผ่านการตรวจสอบ
            # ---------------------------------------------

            if verification_status in SEARCHABLE_INGREDIENT_STATUS:

                expanded_terms = self.search_terms_by_ingredient.get(
                    ingredient_id, []
                )

                for term in expanded_terms:
                    self._add_term(terms, term)

            # ---------------------------------------------
            # 2. Normal searchable ingredients
            # ---------------------------------------------

            if verification_status in SEARCHABLE_INGREDIENT_STATUS:

                candidates = [
                    item["ingredient_name"],
                    item["canonical_standard_name"],
                    catalog.get("true_lexical_synonyms_th", ""),
                    catalog.get("scientific_name_latin", ""),
                    catalog.get("english_name", ""),
                ]

                for candidate in candidates:
                    self._add_term(terms, candidate)

                # -----------------------------------------
                # 2.1 Part-aware Search Expansion
                # -----------------------------------------

                self._add_part_aware_terms(
                    terms,
                    catalog,
                )

            # ---------------------------------------------
            # 3. Verified Ontology Relations
            # ---------------------------------------------

            relations = self.get_verified_relations(
                subject_id=ingredient_id
            )

            for relation in relations:

                predicate = relation["predicate_relation"]
                object_label = relation["object_label"]

                # Same as: เพิ่มชื่อมาตรฐาน
                if predicate == "Same as":
                    self._add_term(
                        terms,
                        object_label,
                    )

                # Decomposed into: ขยายส่วนประกอบของวัตถุดิบ
                elif predicate == "Decomposed into":
                    if self._is_searchable_component(object_label):
                        self._add_term(
                            terms,
                            object_label,
                        )

        return terms


    # =====================================================
    # Summary
    # =====================================================

    def get_summary(self) -> dict[str, int]:
        """สรุปข้อมูลจากโครงสร้าง 3NF และ Derived IR status."""

        verified_relations = self.get_verified_relations()

        status_counts: dict[str, int] = {}

        for row in self.food_ingredients:

            status = row["mapping_status"]

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

        return {
            "catalog_items": len(self.catalog),

            "relations_total": len(self.relations),

            "relations_verified": len(verified_relations),

            "sources": len(self.sources),

            "recipes": len(self.recipes),

            "recipe_ingredient_rows": len(
                self.recipe_ingredients
            ),

            "food_ingredient_rows": len(
                self.food_ingredients
            ),

            "eligible_mapping_rows": (
                status_counts.get("ELIGIBLE", 0)
                + status_counts.get(
                    "ELIGIBLE_WITH_VARIANT_CAVEAT", 0
                )
                + status_counts.get(
                    "ELIGIBLE_VECTOR_DUAL_BRIDGE", 0
                )
            ),

            "eligible_exact_rows": status_counts.get(
                "ELIGIBLE", 0
            ),

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


# =========================================================
# Run Directly
# =========================================================

if __name__ == "__main__":

    ontology = IngredientOntology()

    print("Ingredient Ontology loaded")

    print("\n===== ONTOLOGY SUMMARY =====")

    for key, value in ontology.get_summary().items():
        print(f"{key}: {value}")

    print("\n===== FOOD F027 =====")

    for item in ontology.get_food_ingredients("F027"):

        print(
            item["ingredient_name"],
            "->",
            item["canonical_standard_name"],
            "|",
            item["mapping_status"],
        )

    print("\n===== SEARCH TERMS F027 =====")

    print(
        ontology.get_food_search_terms("F027")
    )
