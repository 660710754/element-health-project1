from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from food_retrieval import retrieve_foods_by_element


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
FOODS_FILE = BASE_DIR / "data" / "foods.csv"

ELEMENTS = (
    "earth",
    "water",
    "wind",
    "fire",
)

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

# คะแนนอันดับ 1 และอันดับ 2 ต่างกันไม่เกิน 1 คะแนน
# ถือว่าเป็น mixed
DEFAULT_MIXED_THRESHOLD = 1.0

# หน้า UI แสดง 4 รายการต่อหมวด
DEFAULT_ITEMS_PER_CATEGORY = 4

DEFAULT_AVOID_LIMIT = 6


# =========================================================
# Load foods.csv
# =========================================================

def load_foods() -> list[dict[str, str]]:
    """
    โหลดและตรวจสอบข้อมูลจาก data/foods.csv
    """

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
        raise ValueError(
            "foods.csv ไม่มีข้อมูล"
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(rows[0].keys())
    )

    if missing_columns:
        raise ValueError(
            "foods.csv ขาดคอลัมน์: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    cleaned_rows: list[dict[str, str]] = []
    seen_food_ids: set[str] = set()

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
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

        if food_id in seen_food_ids:
            raise ValueError(
                f"พบ food_id ซ้ำ: {food_id}"
            )

        seen_food_ids.add(food_id)

        if element not in ELEMENTS:
            raise ValueError(
                f"บรรทัด {line_number} "
                f"ระบุธาตุไม่ถูกต้อง: {element!r}"
            )

        if status not in {
            "recommended",
            "avoid",
        }:
            raise ValueError(
                f"บรรทัด {line_number} "
                "recommendation_status "
                f"ไม่ถูกต้อง: {status!r}"
            )

        cleaned_rows.append(cleaned)

    return cleaned_rows


# =========================================================
# Score validation
# =========================================================

def validate_element_scores(
    scores: dict[str, float],
) -> None:
    """
    ตรวจสอบคะแนนธาตุทั้ง 4
    """

    missing = set(ELEMENTS) - set(scores)

    if missing:
        raise ValueError(
            "คะแนนขาดธาตุ: "
            + ", ".join(sorted(missing))
        )

    unknown = set(scores) - set(ELEMENTS)

    if unknown:
        raise ValueError(
            "พบชื่อธาตุที่ไม่รู้จัก: "
            + ", ".join(sorted(unknown))
        )

    for element in ELEMENTS:
        value = scores[element]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"คะแนน {element} ต้องเป็นตัวเลข"
            )

        if value < 0:
            raise ValueError(
                f"คะแนน {element} ต้องไม่น้อยกว่า 0"
            )


# =========================================================
# Normalize scores
# =========================================================

def normalize_scores(
    scores: dict[str, float],
) -> dict[str, float]:
    """
    แปลงคะแนนทั้ง 4 ธาตุเป็นสัดส่วน
    โดยผลรวมเท่ากับ 1
    """

    validate_element_scores(scores)

    total = sum(
        float(scores[element])
        for element in ELEMENTS
    )

    if total == 0:
        return {
            element: 0.25
            for element in ELEMENTS
        }

    return {
        element: (
            float(scores[element]) / total
        )
        for element in ELEMENTS
    }


# =========================================================
# Rank elements
# =========================================================

def rank_elements(
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    """
    เรียงธาตุจากคะแนนสูงสุดไปต่ำสุด

    หากคะแนนเท่ากัน
    ใช้ลำดับ earth, water, wind, fire
    เพื่อให้ผลลัพธ์คงที่
    """

    validate_element_scores(scores)

    element_order = {
        element: index
        for index, element in enumerate(ELEMENTS)
    }

    ranking = [
        (
            element,
            float(scores[element]),
        )
        for element in ELEMENTS
    ]

    ranking.sort(
        key=lambda item: (
            -item[1],
            element_order[item[0]],
        )
    )

    return ranking


# =========================================================
# Analyze primary / secondary
# =========================================================

def analyze_element_relationship(
    scores: dict[str, float],
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, Any]:
    """
    วิเคราะห์คะแนนอันดับ 1 และอันดับ 2

    equal:
        คะแนนเท่ากัน

    mixed:
        คะแนนต่างกันมากกว่า 0
        แต่ไม่เกิน mixed_threshold

    primary_only:
        คะแนนต่างกันมากกว่า mixed_threshold
    """

    if mixed_threshold < 0:
        raise ValueError(
            "mixed_threshold ต้องไม่น้อยกว่า 0"
        )

    ranking = rank_elements(scores)

    primary_element, primary_score = ranking[0]
    secondary_element, secondary_score = ranking[1]

    difference = (
        primary_score
        - secondary_score
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
        "is_mixed": mode in {
            "equal",
            "mixed",
        },
        "is_primary_only": (
            mode == "primary_only"
        ),
    }


# =========================================================
# Compatibility helper
# =========================================================

def detect_close_elements(
    scores: dict[str, float],
    threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, Any]:
    """
    รองรับโค้ดเดิม

    threshold หมายถึงผลต่างคะแนนจริง
    เช่น 1.0 คะแนน
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
# Calculate quota
# =========================================================

def calculate_element_quotas(
    scores: dict[str, float],
    total_items: int = DEFAULT_ITEMS_PER_CATEGORY,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, int]:
    """
    คำนวณจำนวนรายการของแต่ละธาตุ

    เมื่อ total_items = 4

    equal:
        2 + 2

    mixed:
        3 + 1

    primary_only:
        4 + 0
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

    # Equal → 2+2 เมื่อ total_items = 4
    if mode == "equal":

        secondary_quota = total_items // 2
        primary_quota = (
            total_items - secondary_quota
        )

    # Mixed → 3+1 เมื่อ total_items = 4
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

            primary_quota = (
                total_items
                - secondary_quota
            )

    # Primary only → 4+0
    else:

        primary_quota = total_items
        secondary_quota = 0

    quotas[primary] = primary_quota
    quotas[secondary] = secondary_quota

    return quotas


# =========================================================
# Food score / metadata
# =========================================================

def calculate_food_match_score(
    food: dict[str, str],
    normalized_scores: dict[str, float],
) -> float:
    """
    คะแนนสัดส่วนของธาตุสำหรับแสดงผล

    หมายเหตุ:
    ค่านี้ไม่ใช่ IR cosine similarity
    และไม่ได้ถูกนำไปบวกกับ ir_score
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
    recommendation_mode: str,
) -> dict[str, Any]:
    """
    เพิ่มข้อมูลสำหรับส่งไปแสดงผลใน UI
    """

    result: dict[str, Any] = dict(food)

    element = food["recommended_element"]

    result["match_score"] = (
        calculate_food_match_score(
            food=food,
            normalized_scores=normalized_scores,
        )
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

    result["primary_element"] = (
        primary_element
    )

    result["secondary_element"] = (
        secondary_element
    )

    result["recommendation_mode"] = (
        recommendation_mode
    )

    result["mixed_elements"] = (
        recommendation_mode
        in {
            "equal",
            "mixed",
        }
    )

    if recommendation_mode == "equal":

        result["match_reason"] = (
            f"รายการนี้เป็นอาหารที่แนะนำสำหรับ"
            f"{ELEMENT_NAMES_TH[element]} "
            "เนื่องจากธาตุอันดับหนึ่งและอันดับสอง"
            "มีคะแนนเท่ากัน ระบบจึงแนะนำอาหาร"
            "ของทั้งสองธาตุในสัดส่วนเท่ากัน"
        )

    elif recommendation_mode == "mixed":

        if element == primary_element:

            result["match_reason"] = (
                f"รายการนี้เป็นอาหารที่แนะนำสำหรับ"
                f"{ELEMENT_NAMES_TH[primary_element]} "
                "ซึ่งมีคะแนนสูงที่สุด "
                "จึงได้รับการแนะนำเป็นหลัก"
            )

        else:

            result["match_reason"] = (
                f"รายการนี้เป็นอาหารที่แนะนำสำหรับ"
                f"{ELEMENT_NAMES_TH[secondary_element]} "
                "ซึ่งมีคะแนนใกล้เคียงกับธาตุหลัก "
                "ระบบจึงนำมาประกอบคำแนะนำ"
            )

    else:

        result["match_reason"] = (
            f"รายการนี้เป็นอาหารที่แนะนำสำหรับ"
            f"{ELEMENT_NAMES_TH[primary_element]} "
            "เนื่องจากธาตุหลักมีคะแนนสูงกว่า"
            "ธาตุรองมากกว่าเกณฑ์ที่กำหนด "
            "ระบบจึงใช้รายการของธาตุหลัก"
        )

    return result


# =========================================================
# Duplicate helpers
# =========================================================

def normalize_food_name(
    name: str,
) -> str:
    """
    ปรับชื่ออาหารเพื่อใช้ตรวจชื่อซ้ำ
    """

    return (
        name
        .strip()
        .replace(" ", "")
        .replace("\u200b", "")
    )


def remove_duplicate_food_names(
    foods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    ลบรายการอาหารชื่อซ้ำ
    โดยเก็บรายการแรกไว้
    """

    unique_foods: list[
        dict[str, Any]
    ] = []

    seen_names: set[str] = set()

    for food in foods:

        name = str(
            food.get(
                "food_name_th",
                "",
            )
        )

        normalized_name = (
            normalize_food_name(name)
        )

        if not normalized_name:
            continue

        if normalized_name in seen_names:
            continue

        seen_names.add(
            normalized_name
        )

        unique_foods.append(
            food
        )

    return unique_foods


# =========================================================
# Select items by quota and display order
# =========================================================

def select_items_by_mode(
    primary_items: list[dict[str, Any]],
    secondary_items: list[dict[str, Any]],
    primary_quota: int,
    secondary_quota: int,
    mode: str,
) -> list[dict[str, Any]]:
    """
    เลือกและจัดลำดับรายการให้ตรงกับ UI

    equal:
        P, P, S, S
        เช่น earth, earth, water, water

    mixed:
        P, P, P, S
        เช่น earth, earth, earth, water

    primary_only:
        P, P, P, P
        เช่น earth, earth, earth, earth

    หมายเหตุ:
    ไม่สลับ Primary/Secondary
    เพราะ UI แสดงกลุ่ม Primary ก่อน
    แล้วจึงแสดง Secondary
    """

    selected_primary = (
        primary_items[:primary_quota]
    )

    selected_secondary = (
        secondary_items[:secondary_quota]
    )

    if mode in {
        "equal",
        "mixed",
    }:
        return (
            selected_primary
            + selected_secondary
        )

    return selected_primary


# =========================================================
# Main recommendation
# =========================================================

def recommend_foods(
    scores: dict[str, float],
    limit: int = 10,
    categories: list[str] | None = None,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    แนะนำอาหารจากผลคะแนนธาตุ

    ขั้นตอน:
    1. หา Primary / Secondary
    2. กำหนด quota
    3. เรียก IR ของแต่ละธาตุแยกกัน
    4. กรอง category
    5. เลือกอาหารตามอันดับ IR
    6. จัดสัดส่วนตาม quota

    คะแนนแบบประเมิน:
        ใช้กำหนด Primary / Secondary / Quota

    IR score:
        ใช้จัดอันดับอาหารภายในธาตุ

    ไม่มีการนำคะแนนแบบประเมิน
    ไปบวกกับ cosine similarity
    """

    if limit <= 0:
        raise ValueError(
            "limit ต้องมากกว่า 0"
        )

    normalized = normalize_scores(scores)

    relationship = (
        analyze_element_relationship(
            scores=scores,
            mixed_threshold=mixed_threshold,
        )
    )

    primary = relationship[
        "primary_element"
    ]

    secondary = relationship[
        "secondary_element"
    ]

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

    # primary ต้องใช้เสมอ
    target_elements = [primary]

    # secondary ใช้เฉพาะกรณีมี quota > 0
    if quotas[secondary] > 0:
        target_elements.append(
            secondary
        )

    grouped_results: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    # -----------------------------------------------------
    # Retrieve IR แยกตามธาตุ
    # -----------------------------------------------------

    for element in target_elements:

        element_foods = (
            retrieve_foods_by_element(
                active_elements=[
                    element
                ],
                top_k=50,
            )
        )

        # -------------------------------------------------
        # Category filtering
        # -------------------------------------------------

        if allowed_categories is not None:

            element_foods = [
                food
                for food in element_foods
                if food.get("category")
                in allowed_categories
            ]

        # -------------------------------------------------
        # Prepare metadata
        # -------------------------------------------------

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

        # ลบชื่อซ้ำ
        # แต่ยังรักษาลำดับ IR เดิม
        prepared = (
            remove_duplicate_food_names(
                prepared
            )
        )

        grouped_results[
            element
        ] = prepared

    # -----------------------------------------------------
    # Primary / Secondary candidates
    # -----------------------------------------------------

    primary_items = (
        grouped_results.get(
            primary,
            [],
        )
    )

    secondary_items = (
        grouped_results.get(
            secondary,
            [],
        )
    )

    # -----------------------------------------------------
    # Select by quota
    # -----------------------------------------------------

    selected = select_items_by_mode(
        primary_items=primary_items,
        secondary_items=secondary_items,
        primary_quota=quotas[
            primary
        ],
        secondary_quota=quotas[
            secondary
        ],
        mode=mode,
    )

    selected = (
        remove_duplicate_food_names(
            selected
        )
    )

    # ไม่เติมอาหารจากธาตุอื่น
    # เพื่อรักษา quota 2+2 / 3+1 / 4+0
    #
    # ถ้าฐานข้อมูลของธาตุใดมีรายการไม่ถึง quota
    # ระบบจะคืนเท่าที่มีจริง

    return selected[:limit]


# =========================================================
# Recommendation by category
# =========================================================

def recommend_foods_by_category(
    scores: dict[str, float],
    per_category: int = DEFAULT_ITEMS_PER_CATEGORY,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, list[dict[str, Any]]]:
    """
    แนะนำอาหารแยกตามหมวดที่ UI ใช้

    เมื่อ per_category = 4

    equal:
        2 + 2

    mixed:
        3 + 1

    primary_only:
        4 + 0
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

    result: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for category in categories:

        result[category] = (
            recommend_foods(
                scores=scores,
                limit=per_category,
                categories=[
                    category
                ],
                mixed_threshold=mixed_threshold,
            )
        )

    return result


# =========================================================
# Avoid rules
# =========================================================

def get_avoid_rules(
    scores: dict[str, float],
    limit: int = DEFAULT_AVOID_LIMIT,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> list[dict[str, str]]:
    """
    เลือกข้อควรหลีกเลี่ยง
    ตาม Primary / Secondary และ quota

    เมื่อ limit = 6 โดยประมาณ:

    equal:
        3 + 3

    mixed:
        4 + 2

    primary_only:
        6 + 0

    ลำดับการแสดงผล:
        Primary ก่อน Secondary
    """

    if limit <= 0:
        return []

    foods = load_foods()

    relationship = (
        analyze_element_relationship(
            scores=scores,
            mixed_threshold=mixed_threshold,
        )
    )

    primary = relationship[
        "primary_element"
    ]

    secondary = relationship[
        "secondary_element"
    ]

    mode = relationship["mode"]

    primary_rules = [
        dict(food)
        for food in foods
        if (
            food[
                "recommendation_status"
            ] == "avoid"
            and food[
                "recommended_element"
            ] == primary
        )
    ]

    secondary_rules = [
        dict(food)
        for food in foods
        if (
            food[
                "recommendation_status"
            ] == "avoid"
            and food[
                "recommended_element"
            ] == secondary
        )
    ]

    primary_rules = (
        remove_duplicate_food_names(
            primary_rules
        )
    )

    secondary_rules = (
        remove_duplicate_food_names(
            secondary_rules
        )
    )

    quotas = calculate_element_quotas(
        scores=scores,
        total_items=limit,
        mixed_threshold=mixed_threshold,
    )

    selected = select_items_by_mode(
        primary_items=primary_rules,
        secondary_items=secondary_rules,
        primary_quota=quotas[
            primary
        ],
        secondary_quota=quotas[
            secondary
        ],
        mode=mode,
    )

    selected = (
        remove_duplicate_food_names(
            selected
        )
    )

    return [
        dict(item)
        for item in selected[:limit]
    ]


# =========================================================
# Compatibility wrapper for RAG
# =========================================================

def recommend(
    scores: dict[str, float],
    question: str = "",
    limit: int = 10,
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, Any]:
    """
    Compatibility wrapper สำหรับ rag.py
    """

    result = (
        build_recommendation_summary(
            scores=scores,
            question=question,
            mixed_threshold=mixed_threshold,
        )
    )

    result[
        "top_recommendations"
    ] = result[
        "top_recommendations"
    ][:limit]

    return result


# =========================================================
# Build summary for app.py / rag.py
# =========================================================

def build_recommendation_summary(
    scores: dict[str, float],
    question: str = "",
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> dict[str, Any]:
    """
    สร้างผลสรุปสำหรับ app.py และ rag.py
    """

    if isinstance(
        mixed_threshold,
        str,
    ):
        try:
            mixed_threshold = float(
                mixed_threshold
            )

        except ValueError:
            mixed_threshold = (
                DEFAULT_MIXED_THRESHOLD
            )

    normalized = normalize_scores(
        scores
    )

    relationship = (
        analyze_element_relationship(
            scores=scores,
            mixed_threshold=mixed_threshold,
        )
    )

    primary = relationship[
        "primary_element"
    ]

    secondary = relationship[
        "secondary_element"
    ]

    mode = relationship["mode"]

    difference = relationship[
        "difference"
    ]

    # -----------------------------------------------------
    # 4 รายการต่อหมวด
    # -----------------------------------------------------

    grouped = (
        recommend_foods_by_category(
            scores=scores,
            per_category=(
                DEFAULT_ITEMS_PER_CATEGORY
            ),
            mixed_threshold=mixed_threshold,
        )
    )

    # -----------------------------------------------------
    # Top recommendations
    # -----------------------------------------------------

    top_recommendations = (
        recommend_foods(
            scores=scores,
            limit=10,
            mixed_threshold=mixed_threshold,
        )
    )

    # -----------------------------------------------------
    # Avoid
    # -----------------------------------------------------

    avoid_rules = (
        get_avoid_rules(
            scores=scores,
            limit=DEFAULT_AVOID_LIMIT,
            mixed_threshold=mixed_threshold,
        )
    )

    # -----------------------------------------------------
    # Interpretation
    # -----------------------------------------------------

    if mode == "equal":

        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}และ"
            f"{ELEMENT_NAMES_TH[secondary]}"
            "มีคะแนนเท่ากัน "
            "ระบบจึงแนะนำรายการอาหาร"
            "ของทั้งสองธาตุในสัดส่วนเท่ากัน"
        )

    elif mode == "mixed":

        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}"
            "มีคะแนนสูงที่สุด และ"
            f"{ELEMENT_NAMES_TH[secondary]}"
            f"มีคะแนนต่างกัน {difference:.1f} คะแนน "
            "ซึ่งไม่เกินเกณฑ์ "
            f"{mixed_threshold:.1f} คะแนน "
            "ระบบจึงแนะนำอาหารของธาตุหลัก"
            "เป็นส่วนใหญ่ พร้อมอาหารของธาตุรอง"
        )

    else:

        interpretation = (
            f"{ELEMENT_NAMES_TH[primary]}"
            "มีคะแนนสูงกว่า"
            f"{ELEMENT_NAMES_TH[secondary]} "
            f"{difference:.1f} คะแนน "
            "ซึ่งมากกว่าเกณฑ์ "
            f"{mixed_threshold:.1f} คะแนน "
            "ระบบจึงใช้รายการอาหาร"
            "ของธาตุหลักในการแนะนำ"
        )

    return {
        "primary_element": primary,

        "primary_element_th": (
            ELEMENT_NAMES_TH[
                primary
            ]
        ),

        "secondary_element": secondary,

        "secondary_element_th": (
            ELEMENT_NAMES_TH[
                secondary
            ]
        ),

        "recommendation_mode": mode,

        "is_equal": (
            mode == "equal"
        ),

        # เก็บ key เดิมไว้
        # เพื่อให้ app.py รุ่นเดิมยังทำงาน
        "is_mixed": (
            mode in {
                "equal",
                "mixed",
            }
        ),

        "is_primary_only": (
            mode == "primary_only"
        ),

        "score_difference": (
            difference
        ),

        "relative_difference": (
            relationship[
                "relative_difference"
            ]
        ),

        "mixed_threshold": (
            mixed_threshold
        ),

        "normalized_scores": (
            normalized
        ),

        "interpretation": (
            interpretation
        ),

        "top_recommendations": (
            top_recommendations
        ),

        "recommendations_by_category": (
            grouped
        ),

        "avoid_rules": (
            avoid_rules
        ),

        "question": question,
    }


# =========================================================
# Terminal display
# =========================================================

def print_recommendations(
    scores: dict[str, float],
) -> None:
    """
    แสดงผลสำหรับตรวจสอบใน Terminal
    """

    summary = (
        build_recommendation_summary(
            scores
        )
    )

    mode_names_th = {
        "equal": "คะแนนเท่ากัน",
        "mixed": (
            "ธาตุหลักร่วมกับธาตุรอง"
        ),
        "primary_only": (
            "เฉพาะธาตุหลัก"
        ),
    }

    print("=" * 70)
    print("ผลแนะนำอาหาร")
    print("=" * 70)

    print(
        "ธาตุหลัก:",
        summary[
            "primary_element_th"
        ],
    )

    print(
        "ธาตุรอง:",
        summary[
            "secondary_element_th"
        ],
    )

    print(
        "รูปแบบคำแนะนำ:",
        mode_names_th[
            summary[
                "recommendation_mode"
            ]
        ],
    )

    print(
        "ผลต่างคะแนน:",
        summary[
            "score_difference"
        ],
    )

    print()
    print("สัดส่วนคะแนน")

    for element in ELEMENTS:

        percentage = (
            summary[
                "normalized_scores"
            ][element]
            * 100
        )

        print(
            f"- "
            f"{ELEMENT_NAMES_TH[element]}: "
            f"{percentage:.2f}%"
        )

    print()

    print(
        summary[
            "interpretation"
        ]
    )

    print()
    print(
        "รายการแนะนำแยกตามหมวด"
    )

    for category, foods in (
        summary[
            "recommendations_by_category"
        ].items()
    ):

        print()
        print(
            f"[{CATEGORY_NAMES_TH[category]}]"
        )

        if not foods:
            print(
                "- ไม่มีรายการเพียงพอ"
            )
            continue

        for index, food in enumerate(
            foods,
            start=1,
        ):
            print(
                f"{index}. "
                f"{food['food_name_th']} "
                f"("
                f"{food['recommended_element_th']}"
                f")"
            )

    print()
    print("ข้อควรหลีกเลี่ยง")

    for index, food in enumerate(
        summary[
            "avoid_rules"
        ],
        start=1,
    ):
        print(
            f"{index}. "
            f"{food['food_name_th']} "
            f"("
            f"{food['recommended_element_th']}"
            f")"
        )


# =========================================================
# Manual terminal test
# =========================================================

if __name__ == "__main__":

    # ตัวอย่าง Mixed
    #
    # earth = 10.0
    # water = 9.5
    #
    # Difference = 0.5
    #
    # Expected:
    # earth, earth, earth, water

    sample_scores = {
        "earth": 10.0,
        "water": 9.5,
        "wind": 5.0,
        "fire": 4.0,
    }

    print_recommendations(
        sample_scores
    )