from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


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
    """เรียงธาตุจากคะแนนสูงสุดไปต่ำสุด"""

    validate_element_scores(scores)

    return sorted(
        (
            (element, float(scores[element]))
            for element in ELEMENTS
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def detect_close_elements(
    scores: dict[str, float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    ตรวจว่าธาตุอันดับ 1 และอันดับ 2 มีคะแนนใกล้กันหรือไม่

    threshold = 0.05 หมายถึงต่างกันไม่เกิน 5%
    เมื่อเทียบกับคะแนนของธาตุอันดับหนึ่ง
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold ต้องอยู่ระหว่าง 0 และ 1"
        )

    ranking = rank_elements(scores)

    primary_element, primary_score = ranking[0]
    secondary_element, secondary_score = ranking[1]

    difference = primary_score - secondary_score

    relative_difference = (
        difference / primary_score
        if primary_score > 0
        else 0.0
    )

    return {
        "is_close": relative_difference <= threshold,
        "primary_element": primary_element,
        "primary_score": primary_score,
        "secondary_element": secondary_element,
        "secondary_score": secondary_score,
        "difference": round(difference, 3),
        "relative_difference": round(relative_difference, 4),
        "threshold": threshold,
    }


def calculate_element_quotas(
    scores: dict[str, float],
    total_items: int,
    close_threshold: float = 0.05,
    secondary_share_when_not_close: float = 0.20,
) -> dict[str, int]:
    """
    คำนวณจำนวนอาหารที่ควรเลือกจากแต่ละธาตุ

    กรณีคะแนนใกล้กัน:
    แบ่งจำนวนตามสัดส่วนคะแนนจริงของธาตุอันดับ 1 และ 2

    กรณีคะแนนไม่ใกล้กัน:
    ให้ธาตุเด่นเป็นหลัก และให้ธาตุรองตามสัดส่วนที่กำหนด
    """

    if total_items <= 0:
        raise ValueError("total_items ต้องมากกว่า 0")

    if not 0 <= secondary_share_when_not_close <= 1:
        raise ValueError(
            "secondary_share_when_not_close ต้องอยู่ระหว่าง 0 และ 1"
        )

    close_result = detect_close_elements(
        scores=scores,
        threshold=close_threshold,
    )

    primary = close_result["primary_element"]
    secondary = close_result["secondary_element"]

    quotas = {
        element: 0
        for element in ELEMENTS
    }

    if close_result["is_close"]:
        primary_score = float(scores[primary])
        secondary_score = float(scores[secondary])
        pair_total = primary_score + secondary_score

        primary_ratio = (
            primary_score / pair_total
            if pair_total > 0
            else 0.5
        )

        primary_quota = round(total_items * primary_ratio)

        if total_items >= 2:
            primary_quota = max(
                1,
                min(primary_quota, total_items - 1),
            )

        secondary_quota = total_items - primary_quota

    else:
        secondary_quota = round(
            total_items * secondary_share_when_not_close
        )

        if total_items >= 2:
            secondary_quota = max(1, secondary_quota)

        secondary_quota = min(
            secondary_quota,
            total_items,
        )

        primary_quota = total_items - secondary_quota

    quotas[primary] = primary_quota
    quotas[secondary] = secondary_quota

    return quotas


def calculate_food_match_score(
    food: dict[str, str],
    normalized_scores: dict[str, float],
) -> float:
    """
    คะแนนอาหารใช้สัดส่วนคะแนนจริงของธาตุเท่านั้น

    ไม่มีโบนัสคงที่ เช่น 18, 16, 25 หรือ 8
    """

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
    is_mixed: bool,
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
    result["mixed_elements"] = is_mixed

    if is_mixed:
        result["match_reason"] = (
            f"รายการนี้อยู่ในกลุ่มอาหารที่เอกสารแนะนำสำหรับ "
            f"{ELEMENT_NAMES_TH[element]} "
            "ซึ่งเป็นหนึ่งในสองธาตุที่มีคะแนนใกล้เคียงกัน"
        )
    elif element == primary_element:
        result["match_reason"] = (
            f"รายการนี้อยู่ในกลุ่มอาหารที่เอกสารแนะนำสำหรับ "
            f"{ELEMENT_NAMES_TH[primary_element]} "
            "ซึ่งเป็นธาตุเด่นปัจจุบันของผู้ใช้"
        )
    else:
        result["match_reason"] = (
            f"รายการนี้อยู่ในกลุ่มอาหารที่เอกสารแนะนำสำหรับ "
            f"{ELEMENT_NAMES_TH[secondary_element]} "
            "ซึ่งเป็นธาตุรองของผู้ใช้"
        )

    return result


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


def interleave_foods(
    grouped_foods: dict[str, list[dict[str, Any]]],
    element_order: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """สลับรายการของแต่ละธาตุ เช่น ดิน, น้ำ, ดิน, น้ำ"""

    result: list[dict[str, Any]] = []
    positions = {
        element: 0
        for element in element_order
    }

    while len(result) < limit:
        added = False

        for element in element_order:
            position = positions[element]
            foods = grouped_foods.get(element, [])

            if position < len(foods):
                result.append(foods[position])
                positions[element] += 1
                added = True

                if len(result) >= limit:
                    break

        if not added:
            break

    return result


def recommend_foods(
    scores: dict[str, float],
    limit: int = 10,
    categories: list[str] | None = None,
    close_threshold: float = 0.05,
    secondary_share_when_not_close: float = 0.20,
) -> list[dict[str, Any]]:
    """
    แนะนำอาหารโดยใช้คะแนนจริงจากแบบประเมิน

    ขั้นตอน:
    1. Normalize คะแนนธาตุ
    2. ตรวจว่าธาตุอันดับ 1 และ 2 ใกล้กันหรือไม่
    3. คำนวณโควตาตามสัดส่วนคะแนนจริง
    4. เลือกอาหารจากธาตุเด่นและธาตุรอง
    5. สลับรายการและลบชื่อซ้ำ
    """

    if limit <= 0:
        raise ValueError("limit ต้องมากกว่า 0")

    foods = load_foods()
    normalized = normalize_scores(scores)

    close_result = detect_close_elements(
        scores=scores,
        threshold=close_threshold,
    )

    primary = close_result["primary_element"]
    secondary = close_result["secondary_element"]
    is_mixed = bool(close_result["is_close"])

    quotas = calculate_element_quotas(
        scores=scores,
        total_items=limit,
        close_threshold=close_threshold,
        secondary_share_when_not_close=(
            secondary_share_when_not_close
        ),
    )

    allowed_categories = (
        set(categories)
        if categories is not None
        else None
    )

    grouped_results: dict[str, list[dict[str, Any]]] = {}
    reserve_multiplier = 3

    for element in (primary, secondary):
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
                is_mixed=is_mixed,
            )
            for food in element_foods
        ]

        # เอกสารไม่ได้จัดอันดับอาหารภายในธาตุเดียวกัน
        # จึงเรียงชื่อเพื่อให้ผลลัพธ์คงที่และตรวจสอบซ้ำได้
        prepared.sort(
            key=lambda item: normalize_food_name(
                item["food_name_th"]
            )
        )

        grouped_results[element] = prepared[
            : max(
                quotas[element] * reserve_multiplier,
                quotas[element],
            )
        ]

    interleaved = interleave_foods(
        grouped_foods=grouped_results,
        element_order=[primary, secondary],
        limit=limit * reserve_multiplier,
    )

    unique_results = remove_duplicate_food_names(
        interleaved
    )

    selected = unique_results[:limit]

    # หากลบชื่อซ้ำแล้วจำนวนยังไม่ครบ ให้เติมจากรายการสำรอง
    if len(selected) < limit:
        existing_names = {
            normalize_food_name(item["food_name_th"])
            for item in selected
        }

        remaining_candidates: list[dict[str, Any]] = []

        for element in (primary, secondary):
            for item in grouped_results.get(element, []):
                name = normalize_food_name(
                    item["food_name_th"]
                )

                if name not in existing_names:
                    remaining_candidates.append(item)
                    existing_names.add(name)

        selected.extend(
            remaining_candidates[
                : limit - len(selected)
            ]
        )

    return selected[:limit]


def recommend_foods_by_category(
    scores: dict[str, float],
    per_category: int = 4,
    close_threshold: float = 0.05,
    secondary_share_when_not_close: float = 0.20,
) -> dict[str, list[dict[str, Any]]]:
    """
    แนะนำอาหารแยกตามหมวด

    หากสองธาตุใกล้กันและ per_category = 4
    ระบบจะกระจายรายการตามสัดส่วนคะแนนจริง
    """

    if per_category <= 0:
        raise ValueError("per_category ต้องมากกว่า 0")

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
            close_threshold=close_threshold,
            secondary_share_when_not_close=(
                secondary_share_when_not_close
            ),
        )
        for category in categories
    }


def get_avoid_rules(
    scores: dict[str, float],
    close_threshold: float = 0.05,
) -> list[dict[str, str]]:
    """ดึงข้อควรหลีกเลี่ยงของธาตุเด่นและธาตุรอง"""

    foods = load_foods()

    close_result = detect_close_elements(
        scores=scores,
        threshold=close_threshold,
    )

    primary = close_result["primary_element"]
    secondary = close_result["secondary_element"]

    target_elements = {
        primary,
        secondary,
    }

    avoid_rules = [
        food
        for food in foods
        if (
            food["recommendation_status"] == "avoid"
            and food["recommended_element"] in target_elements
        )
    ]

    avoid_rules.sort(
        key=lambda item: (
            item["recommended_element"] != primary,
            normalize_food_name(item["food_name_th"]),
        )
    )

    return [
        dict(item)
        for item in remove_duplicate_food_names(
            [dict(item) for item in avoid_rules]
        )
    ]


def build_recommendation_summary(
    scores: dict[str, float],
    close_threshold: float = 0.05,
) -> dict[str, Any]:
    """สร้างผลสรุปสำหรับแสดงบนหน้าเว็บ"""

    normalized = normalize_scores(scores)

    close_result = detect_close_elements(
        scores=scores,
        threshold=close_threshold,
    )

    primary = close_result["primary_element"]
    secondary = close_result["secondary_element"]

    grouped = recommend_foods_by_category(
        scores=scores,
        per_category=4,
        close_threshold=close_threshold,
    )

    top_recommendations = recommend_foods(
        scores=scores,
        limit=10,
        close_threshold=close_threshold,
    )

    avoid_rules = get_avoid_rules(
        scores=scores,
        close_threshold=close_threshold,
    )

    if close_result["is_close"]:
        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}และ"
            f"{ELEMENT_NAMES_TH[secondary]}"
            "มีคะแนนใกล้เคียงกัน "
            "ระบบจึงกระจายตัวอย่างอาหารของทั้งสองธาตุ "
            "ตามสัดส่วนคะแนนจากแบบประเมิน"
        )
    else:
        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}"
            "เป็นธาตุเด่นปัจจุบัน "
            f"โดยใช้{ELEMENT_NAMES_TH[secondary]}"
            "เป็นข้อมูลประกอบ"
        )

    return {
        "primary_element": primary,
        "primary_element_th": ELEMENT_NAMES_TH[primary],
        "secondary_element": secondary,
        "secondary_element_th": ELEMENT_NAMES_TH[secondary],
        "is_mixed": close_result["is_close"],
        "score_difference": close_result["difference"],
        "relative_difference": close_result["relative_difference"],
        "normalized_scores": normalized,
        "interpretation": interpretation,
        "top_recommendations": top_recommendations,
        "recommendations_by_category": grouped,
        "avoid_rules": avoid_rules,
    }


def print_recommendations(
    scores: dict[str, float],
) -> None:
    """พิมพ์ผลสำหรับตรวจสอบใน Terminal"""

    summary = build_recommendation_summary(scores)

    print("=" * 60)
    print("ผลแนะนำอาหาร")
    print("=" * 60)

    print("ธาตุเด่น:", summary["primary_element_th"])
    print("ธาตุรอง:", summary["secondary_element_th"])
    print(
        "ธาตุเด่นร่วมกัน:",
        "ใช่" if summary["is_mixed"] else "ไม่ใช่",
    )

    print()
    print("สัดส่วนคะแนน")

    for element in ELEMENTS:
        percentage = summary["normalized_scores"][element] * 100
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


if __name__ == "__main__":
    sample_scores = {
        "earth": 15.0,
        "water": 14.6,
        "wind": 13.1,
        "fire": 13.6,
    }

    print_recommendations(sample_scores)
