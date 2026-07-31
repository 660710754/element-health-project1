from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any


# =========================================================
# ค่าพื้นฐาน
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
FOODS_FILE = BASE_DIR / "data" / "foods.csv"

ELEMENTS = ("earth", "water", "wind", "fire")

ELEMENT_NAMES_TH = {
    "earth": "ธาตุดิน",
    "water": "ธาตุน้ำ",
    "wind": "ธาตุลม",
    "fire": "ธาตุไฟ",
}

CATEGORY_NAMES_TH = {
    "menu": "เมนูอาหาร",
    "vegetable_herb": "ผักพื้นบ้านและสมุนไพร",
    "fruit": "ผลไม้",
    "snack": "อาหารว่าง",
    "drink": "เครื่องดื่ม",
    "avoid_rule": "ข้อควรหลีกเลี่ยง",
}

REQUIRED_COLUMNS = {
    "food_id",
    "food_name_th",
    "category",
    "category_th",
    "recommended_element",
    "recommended_element_th",
    "taste_profile",
    "recommendation_status",
    "reason_th",
}

DEFAULT_MIXED_THRESHOLD = 3.0
DEFAULT_ITEMS_PER_CATEGORY = 4
DEFAULT_AVOID_LIMIT = 6


# =========================================================
# โหลดและตรวจสอบ foods.csv
# =========================================================

def load_foods() -> list[dict[str, str]]:
    """โหลดและตรวจสอบฐานข้อมูลอาหารจาก data/foods.csv"""

    if not FOODS_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ฐานข้อมูลอาหาร: {FOODS_FILE}"
        )

    with FOODS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("foods.csv ไม่มีข้อมูลอาหาร")

    missing_columns = REQUIRED_COLUMNS - set(rows[0].keys())

    if missing_columns:
        raise ValueError(
            "foods.csv ขาดคอลัมน์: "
            + ", ".join(sorted(missing_columns))
        )

    cleaned_rows: list[dict[str, str]] = []

    for line_number, row in enumerate(rows, start=2):
        cleaned = {
            key: (value or "").strip()
            for key, value in row.items()
        }

        food_id = cleaned["food_id"]
        element = cleaned["recommended_element"]
        status = cleaned["recommendation_status"]

        if not food_id:
            raise ValueError(
                f"บรรทัด {line_number} ไม่มี food_id"
            )

        if element not in ELEMENTS:
            raise ValueError(
                f"บรรทัด {line_number} ระบุธาตุไม่ถูกต้อง: "
                f"{element!r}"
            )

        if status not in {"recommended", "avoid"}:
            raise ValueError(
                f"บรรทัด {line_number} ระบุสถานะไม่ถูกต้อง: "
                f"{status!r}"
            )

        cleaned_rows.append(cleaned)

    return cleaned_rows


# =========================================================
# ตรวจสอบและจัดการคะแนนธาตุ
# =========================================================

def validate_element_scores(
    scores: dict[str, float],
) -> None:
    """ตรวจสอบว่าคะแนนธาตุทั้ง 4 ครบและเป็นค่าที่ถูกต้อง"""

    missing = set(ELEMENTS) - set(scores)

    if missing:
        raise ValueError(
            "คะแนนขาดธาตุ: "
            + ", ".join(sorted(missing))
        )

    for element in ELEMENTS:
        value = scores[element]

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"คะแนน {element} ต้องเป็นตัวเลข"
            )

        if value < 0:
            raise ValueError(
                f"คะแนน {element} ต้องไม่น้อยกว่า 0"
            )


def normalize_scores(
    scores: dict[str, float],
) -> dict[str, float]:
    """แปลงคะแนนทั้ง 4 ธาตุเป็นสัดส่วนที่รวมกันเท่ากับ 1"""

    validate_element_scores(scores)

    total = sum(float(scores[element]) for element in ELEMENTS)

    if total == 0:
        return {
            element: 0.25
            for element in ELEMENTS
        }

    return {
        element: float(scores[element]) / total
        for element in ELEMENTS
    }


def rank_elements(
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    """
    เรียงธาตุจากคะแนนสูงสุดไปต่ำสุด

    หากคะแนนเท่ากัน จะเรียงตามลำดับใน ELEMENTS
    เพื่อให้ผลลัพธ์คงที่ทุกครั้ง
    """

    validate_element_scores(scores)

    element_order = {
        element: index
        for index, element in enumerate(ELEMENTS)
    }

    return sorted(
        (
            (element, float(scores[element]))
            for element in ELEMENTS
        ),
        key=lambda item: (
            -item[1],
            element_order[item[0]],
        ),
    )


# =========================================================
# วิเคราะห์ความสัมพันธ์ของธาตุหลักและธาตุรอง
# =========================================================

def analyze_element_relationship(
    scores: dict[str, float],
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, Any]:
    """
    วิเคราะห์ความสัมพันธ์ระหว่างธาตุอันดับ 1 และอันดับ 2

    mode:
    - equal: คะแนนเท่ากัน
    - mixed: คะแนนต่างกันมากกว่า 0 แต่ไม่เกิน 3 คะแนน
    - primary_only: คะแนนต่างกันมากกว่า 3 คะแนน
    """

    if mixed_threshold < 0:
        raise ValueError(
            "mixed_threshold ต้องไม่น้อยกว่า 0"
        )

    ranking = rank_elements(scores)

    primary_element, primary_score = ranking[0]
    secondary_element, secondary_score = ranking[1]

    difference = round(
        primary_score - secondary_score,
        6,
    )

    if abs(difference) < 1e-9:
        mode = "equal"
    elif difference <= mixed_threshold:
        mode = "mixed"
    else:
        mode = "primary_only"

    relative_difference = (
        difference / primary_score
        if primary_score > 0
        else 0.0
    )

    return {
        "mode": mode,
        "primary_element": primary_element,
        "primary_score": primary_score,
        "secondary_element": secondary_element,
        "secondary_score": secondary_score,
        "difference": round(difference, 3),
        "relative_difference": round(
            relative_difference,
            4,
        ),
        "mixed_threshold": mixed_threshold,
        "is_equal": mode == "equal",
        "is_mixed": mode in {"equal", "mixed"},
        "is_primary_only": mode == "primary_only",
    }


def detect_close_elements(
    scores: dict[str, float],
    threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, Any]:
    """
    ฟังก์ชันรองรับโค้ดเดิม

    threshold ปัจจุบันหมายถึง "จำนวนคะแนน"
    เช่น 3.0 คะแนน ไม่ใช่ร้อยละ
    """

    result = analyze_element_relationship(
        scores=scores,
        mixed_threshold=threshold,
    )

    return {
        **result,
        "is_close": result["is_mixed"],
        "threshold": threshold,
    }


# =========================================================
# คำนวณจำนวนรายการของแต่ละธาตุ
# =========================================================

def calculate_element_quotas(
    scores: dict[str, float],
    total_items: int,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, int]:
    """
    คำนวณจำนวนรายการของแต่ละธาตุ

    เมื่อ total_items = 4:
    - คะแนนเท่ากัน -> หลัก 2 + รอง 2
    - ต่างกัน 0–3 คะแนน -> หลัก 3 + รอง 1
    - ต่างกันมากกว่า 3 คะแนน -> หลัก 4 + รอง 0
    """

    if total_items <= 0:
        raise ValueError(
            "total_items ต้องมากกว่า 0"
        )

    relationship = analyze_element_relationship(
        scores=scores,
        mixed_threshold=mixed_threshold,
    )

    primary = relationship["primary_element"]
    secondary = relationship["secondary_element"]
    mode = relationship["mode"]

    quotas = {
        element: 0
        for element in ELEMENTS
    }

    if mode == "equal":
        secondary_quota = total_items // 2
        primary_quota = total_items - secondary_quota

    elif mode == "mixed":
        if total_items == 1:
            primary_quota = 1
            secondary_quota = 0
        else:
            secondary_quota = max(
                1,
                round(total_items * 0.25),
            )
            secondary_quota = min(
                secondary_quota,
                total_items - 1,
            )
            primary_quota = total_items - secondary_quota

    else:
        primary_quota = total_items
        secondary_quota = 0

    quotas[primary] = primary_quota
    quotas[secondary] = secondary_quota

    return quotas


# =========================================================
# เตรียมข้อมูลอาหาร
# =========================================================

def calculate_food_match_score(
    food: dict[str, str],
    normalized_scores: dict[str, float],
) -> float:
    """คำนวณคะแนนความสอดคล้องจากสัดส่วนคะแนนธาตุจริง"""

    element = food["recommended_element"]

    return round(
        normalized_scores[element] * 100.0,
        3,
    )


def prepare_food_result(
    food: dict[str, str],
    normalized_scores: dict[str, float],
    primary_element: str,
    secondary_element: str,
    recommendation_mode: str,
) -> dict[str, Any]:
    """เตรียมข้อมูลอาหารสำหรับส่งไปแสดงผล"""

    result: dict[str, Any] = dict(food)

    element = food["recommended_element"]

    result["match_score"] = calculate_food_match_score(
        food=food,
        normalized_scores=normalized_scores,
    )

    result["element_proportion"] = round(
        normalized_scores[element],
        4,
    )

    result["is_primary_match"] = (
        element == primary_element
    )

    result["is_secondary_match"] = (
        element == secondary_element
    )

    result["primary_element"] = primary_element
    result["secondary_element"] = secondary_element
    result["recommendation_mode"] = recommendation_mode
    result["mixed_elements"] = (
        recommendation_mode in {"equal", "mixed"}
    )
    
    if recommendation_mode == "equal":
       result["match_reason"] = (
        f"รายการนี้เป็นอาหารที่แนะนำสำหรับ{ELEMENT_NAMES_TH[element]} "
        "เนื่องจากธาตุหลักและธาตุรองมีคะแนนเท่ากัน "
        "ระบบจึงแนะนำข้อมูลของทั้งสองธาตุในสัดส่วนที่เท่ากัน"
    )

    elif recommendation_mode == "mixed":
        if element == primary_element:
           result["match_reason"] = (
            f"รายการนี้เป็นอาหารที่แนะนำสำหรับ{ELEMENT_NAMES_TH[primary_element]} "
            "ซึ่งเป็นธาตุที่มีคะแนนสูงที่สุด "
            "จึงได้รับการแนะนำเป็นหลัก"
        )
        else:
            result["match_reason"] = (
            f"รายการนี้เป็นอาหารที่แนะนำสำหรับ{ELEMENT_NAMES_TH[secondary_element]} "
            "ซึ่งมีคะแนนใกล้เคียงกับธาตุหลัก "
            "ระบบจึงแสดงเป็นข้อมูลประกอบการแนะนำ"
        )

    else:
        result["match_reason"] = (
        f"รายการนี้เป็นอาหารที่แนะนำสำหรับ{ELEMENT_NAMES_TH[primary_element]} "
        "เนื่องจากมีคะแนนสูงกว่าธาตุอื่นอย่างชัดเจน "
        "ระบบจึงแนะนำเฉพาะข้อมูลของธาตุนี้"
    )

    return result

# =========================================================
# ฟังก์ชันช่วยลบชื่อซ้ำและเลือกตามโควตา
# =========================================================

def normalize_food_name(name: str) -> str:
    """ปรับชื่ออาหารเพื่อใช้ตรวจชื่อซ้ำ"""

    return (
        name
        .strip()
        .replace(" ", "")
        .replace("\u200b", "")
    )


def remove_duplicate_food_names(
    foods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """ลบรายการอาหารชื่อซ้ำ โดยเก็บรายการแรกไว้"""

    unique_foods: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for food in foods:
        normalized_name = normalize_food_name(
            food["food_name_th"]
        )

        if normalized_name in seen_names:
            continue

        seen_names.add(normalized_name)
        unique_foods.append(food)

    return unique_foods


def interleave_two_groups(
    primary_items: list[dict[str, Any]],
    secondary_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """สลับรายการธาตุหลักและธาตุรอง"""

    result: list[dict[str, Any]] = []
    max_length = max(
        len(primary_items),
        len(secondary_items),
    )

    for index in range(max_length):
        if index < len(primary_items):
            result.append(primary_items[index])

        if index < len(secondary_items):
            result.append(secondary_items[index])

    return result


def select_items_by_mode(
    primary_items: list[dict[str, Any]],
    secondary_items: list[dict[str, Any]],
    primary_quota: int,
    secondary_quota: int,
    mode: str,
) -> list[dict[str, Any]]:
    """เลือกและจัดลำดับรายการตามโหมดคำแนะนำ"""

    selected_primary = primary_items[:primary_quota]
    selected_secondary = secondary_items[:secondary_quota]

    if mode == "equal":
        return interleave_two_groups(
            selected_primary,
            selected_secondary,
        )

    if mode == "mixed":
        # แสดงรายการธาตุหลักทั้งหมดก่อน แล้วจึงตามด้วยธาตุรอง
        # ตัวอย่าง 4 รายการ: หลัก, หลัก, หลัก, รอง
        return selected_primary + selected_secondary

    return selected_primary


# =========================================================
# ระบบแนะนำอาหาร
# =========================================================

def recommend_foods(
    scores: dict[str, float],
    limit: int = 10,
    categories: list[str] | None = None,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    แนะนำอาหารตามความต่างของคะแนนธาตุ

    - equal:
      ธาตุหลักและธาตุรองเท่ากัน

    - mixed:
      ธาตุหลักเป็นส่วนใหญ่และสอดแทรกธาตุรอง

    - primary_only:
      เฉพาะธาตุหลัก
    """

    if limit <= 0:
        raise ValueError(
            "limit ต้องมากกว่า 0"
        )

    foods = load_foods()
    normalized = normalize_scores(scores)

    relationship = analyze_element_relationship(
        scores=scores,
        mixed_threshold=mixed_threshold,
    )

    primary = relationship["primary_element"]
    secondary = relationship["secondary_element"]
    mode = relationship["mode"]

    quotas = calculate_element_quotas(
        scores=scores,
        total_items=limit,
        mixed_threshold=mixed_threshold,
    )

    allowed_categories = (
        set(categories)
        if categories is not None
        else None
    )

    target_elements = [primary]

    if quotas[secondary] > 0:
        target_elements.append(secondary)

    grouped_results: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for element in target_elements:
        element_foods = [
            food
            for food in foods
            if (
                food["recommendation_status"] == "recommended"
                and food["recommended_element"] == element
                and (
                    allowed_categories is None
                    or food["category"] in allowed_categories
                )
            )
        ]

        prepared = [
            prepare_food_result(
                food=food,
                normalized_scores=normalized,
                primary_element=primary,
                secondary_element=secondary,
                recommendation_mode=mode,
            )
            for food in element_foods
        ]

        # ลบชื่อซ้ำก่อน แล้วสุ่มลำดับรายการภายในธาตุ
        # เพื่อไม่ให้ผลลัพธ์เรียงตามตัวอักษรทุกครั้ง
        prepared = remove_duplicate_food_names(prepared)
        random.shuffle(prepared)

        grouped_results[element] = prepared

    primary_items = grouped_results.get(primary, [])
    secondary_items = grouped_results.get(secondary, [])

    selected = select_items_by_mode(
        primary_items=primary_items,
        secondary_items=secondary_items,
        primary_quota=quotas[primary],
        secondary_quota=quotas[secondary],
        mode=mode,
    )

    selected = remove_duplicate_food_names(selected)

    # เติมรายการให้ครบเฉพาะจากธาตุที่อนุญาตในโหมดนั้น
    if len(selected) < limit:
        existing_names = {
            normalize_food_name(item["food_name_th"])
            for item in selected
        }

        fallback_candidates = (
            primary_items + secondary_items
        )

        for item in fallback_candidates:
            name = normalize_food_name(
                item["food_name_th"]
            )

            if name in existing_names:
                continue

            selected.append(item)
            existing_names.add(name)

            if len(selected) >= limit:
                break

    return selected[:limit]


# =========================================================
# แนะนำอาหารแยกตามหมวด
# =========================================================

def recommend_foods_by_category(
    scores: dict[str, float],
    per_category: int = DEFAULT_ITEMS_PER_CATEGORY,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, list[dict[str, Any]]]:
    """
    แนะนำอาหารแยกตามหมวด

    เมื่อ per_category = 4:
    - คะแนนเท่ากัน: หลัก 2 + รอง 2
    - ต่างกันไม่เกิน 3: หลัก 3 + รอง 1
    - ต่างกันมากกว่า 3: หลัก 4
    """

    if per_category <= 0:
        raise ValueError(
            "per_category ต้องมากกว่า 0"
        )

    categories = [
        "menu",
        "vegetable_herb",
        "fruit",
        "snack",
        "drink",
    ]

    return {
        category: recommend_foods(
            scores=scores,
            limit=per_category,
            categories=[category],
            mixed_threshold=mixed_threshold,
        )
        for category in categories
    }


# =========================================================
# อาหารที่ควรระวัง
# =========================================================

def get_avoid_rules(
    scores: dict[str, float],
    limit: int = DEFAULT_AVOID_LIMIT,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> list[dict[str, str]]:
    """
    เลือกอาหารที่ควรระวังตามเงื่อนไขเดียวกับอาหารแนะนำ

    เมื่อ limit = 6:
    - คะแนนเท่ากัน: หลัก 3 + รอง 3
    - ต่างกันไม่เกิน 3: หลักประมาณ 4–5 + รอง 1–2
    - ต่างกันมากกว่า 3: หลัก 6
    """

    if limit <= 0:
        return []

    foods = load_foods()

    relationship = analyze_element_relationship(
        scores=scores,
        mixed_threshold=mixed_threshold,
    )

    primary = relationship["primary_element"]
    secondary = relationship["secondary_element"]
    mode = relationship["mode"]

    primary_rules = [
        dict(food)
        for food in foods
        if (
            food["recommendation_status"] == "avoid"
            and food["recommended_element"] == primary
        )
    ]

    secondary_rules = [
        dict(food)
        for food in foods
        if (
            food["recommendation_status"] == "avoid"
            and food["recommended_element"] == secondary
        )
    ]

    primary_rules.sort(
        key=lambda item: normalize_food_name(
            item["food_name_th"]
        )
    )

    secondary_rules.sort(
        key=lambda item: normalize_food_name(
            item["food_name_th"]
        )
    )

    primary_rules = remove_duplicate_food_names(
        primary_rules
    )

    secondary_rules = remove_duplicate_food_names(
        secondary_rules
    )

    quotas = calculate_element_quotas(
        scores=scores,
        total_items=limit,
        mixed_threshold=mixed_threshold,
    )

    selected = select_items_by_mode(
        primary_items=primary_rules,
        secondary_items=secondary_rules,
        primary_quota=quotas[primary],
        secondary_quota=quotas[secondary],
        mode=mode,
    )

    selected = remove_duplicate_food_names(selected)

    if len(selected) < limit:
        existing_names = {
            normalize_food_name(item["food_name_th"])
            for item in selected
        }

        fallback_candidates = (
            primary_rules + secondary_rules
        )

        for item in fallback_candidates:
            name = normalize_food_name(
                item["food_name_th"]
            )

            if name in existing_names:
                continue

            selected.append(item)
            existing_names.add(name)

            if len(selected) >= limit:
                break

    return [
        dict(item)
        for item in selected[:limit]
    ]


# =========================================================
# สรุปผลสำหรับ app.py
# =========================================================

def build_recommendation_summary(
    scores: dict[str, float],
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, Any]:
    """สร้างผลสรุปสำหรับแสดงบนหน้าเว็บ"""

    normalized = normalize_scores(scores)

    relationship = analyze_element_relationship(
        scores=scores,
        mixed_threshold=mixed_threshold,
    )

    primary = relationship["primary_element"]
    secondary = relationship["secondary_element"]
    mode = relationship["mode"]
    difference = relationship["difference"]

    grouped = recommend_foods_by_category(
        scores=scores,
        per_category=DEFAULT_ITEMS_PER_CATEGORY,
        mixed_threshold=mixed_threshold,
    )

    top_recommendations = recommend_foods(
        scores=scores,
        limit=10,
        mixed_threshold=mixed_threshold,
    )

    avoid_rules = get_avoid_rules(
        scores=scores,
        limit=DEFAULT_AVOID_LIMIT,
        mixed_threshold=mixed_threshold,
    )

    if mode == "equal":
        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}และ"
            f"{ELEMENT_NAMES_TH[secondary]}"
            "มีคะแนนเท่ากัน "
            "ระบบจึงแสดงตัวอย่างอาหารและข้อควรระวัง "
            "ของทั้งสองธาตุในจำนวนเท่ากัน"
        )

    elif mode == "mixed":
        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}"
            "มีคะแนนสูงที่สุด และ"
            f"{ELEMENT_NAMES_TH[secondary]}"
            f"มีคะแนนต่างกัน {difference:.1f} คะแนน "
            "ระบบจึงแนะนำข้อมูลของธาตุหลักเป็นส่วนใหญ่ "
            "พร้อมสอดแทรกข้อมูลของธาตุรอง"
        )

    else:
        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}"
            f"มีคะแนนสูงกว่า"
            f"{ELEMENT_NAMES_TH[secondary]} "
            f"{difference:.1f} คะแนน "
            "ระบบจึงแนะนำอาหารและข้อควรระวัง "
            "ของธาตุหลักเท่านั้น"
        )

    return {
        "primary_element": primary,
        "primary_element_th": ELEMENT_NAMES_TH[primary],
        "secondary_element": secondary,
        "secondary_element_th": ELEMENT_NAMES_TH[
            secondary
        ],
        "recommendation_mode": mode,
        "is_equal": mode == "equal",
        # คง key เดิมไว้เพื่อให้ app.py รุ่นเดิมยังใช้งานได้
        "is_mixed": mode in {"equal", "mixed"},
        "is_primary_only": mode == "primary_only",
        "score_difference": difference,
        "relative_difference": relationship[
            "relative_difference"
        ],
        "mixed_threshold": mixed_threshold,
        "normalized_scores": normalized,
        "interpretation": interpretation,
        "top_recommendations": top_recommendations,
        "recommendations_by_category": grouped,
        "avoid_rules": avoid_rules,
    }


# =========================================================
# แสดงผลใน Terminal
# =========================================================

def print_recommendations(
    scores: dict[str, float],
) -> None:
    """พิมพ์ผลสำหรับตรวจสอบใน Terminal"""

    summary = build_recommendation_summary(scores)

    mode_names_th = {
        "equal": "คะแนนเท่ากัน",
        "mixed": "ธาตุหลักร่วมกับธาตุรอง",
        "primary_only": "เฉพาะธาตุหลัก",
    }

    print("=" * 60)
    print("ผลแนะนำอาหาร")
    print("=" * 60)

    print(
        "ธาตุหลัก:",
        summary["primary_element_th"],
    )
    print(
        "ธาตุรอง:",
        summary["secondary_element_th"],
    )
    print(
        "รูปแบบคำแนะนำ:",
        mode_names_th[
            summary["recommendation_mode"]
        ],
    )
    print(
        "ผลต่างคะแนน:",
        summary["score_difference"],
    )

    print()
    print("สัดส่วนคะแนน")

    for element in ELEMENTS:
        percentage = (
            summary["normalized_scores"][element]
            * 100
        )

        print(
            f"- {ELEMENT_NAMES_TH[element]}: "
            f"{percentage:.2f}%"
        )

    print()
    print(summary["interpretation"])

    print()
    print("ตัวอย่างอาหารที่เอกสารแนะนำ")

    for index, food in enumerate(
        summary["top_recommendations"],
        start=1,
    ):
        print(
            f"{index}. {food['food_name_th']} "
            f"({food['recommended_element_th']})"
        )

    print()
    print("อาหารที่ควรระวัง")

    for index, food in enumerate(
        summary["avoid_rules"],
        start=1,
    ):
        print(
            f"{index}. {food['food_name_th']} "
            f"({food['recommended_element_th']})"
        )


# =========================================================
# ทดสอบเมื่อรันไฟล์โดยตรง
# =========================================================

if __name__ == "__main__":
    # ตัวอย่างกรณีต่างกัน 0.5 คะแนน
    # ควรได้โหมด mixed และหมวดละ 3:1
    sample_scores = {
        "earth": 16.1,
        "water": 15.5,
        "wind": 12.1,
        "fire": 15.6,
    }

    print_recommendations(sample_scores)
