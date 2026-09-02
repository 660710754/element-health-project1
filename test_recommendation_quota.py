from __future__ import annotations

from collections import Counter
from typing import Any

from recommendation import (
    DEFAULT_ITEMS_PER_CATEGORY,
    DEFAULT_MIXED_THRESHOLD,
    analyze_element_relationship,
    calculate_element_quotas,
    recommend_foods_by_category,
)


# =========================================================
# Configuration
# =========================================================

PER_CATEGORY = DEFAULT_ITEMS_PER_CATEGORY
THRESHOLD = DEFAULT_MIXED_THRESHOLD

CATEGORIES = (
    "menu",
    "vegetable_herb",
    "fruit",
    "snack",
    "drink",
)


# =========================================================
# Helper Functions
# =========================================================

def count_elements(
    foods: list[dict[str, Any]],
) -> Counter:
    """
    นับจำนวนอาหารของแต่ละธาตุ
    """
    return Counter(
        food["recommended_element"]
        for food in foods
    )


def get_element_order(
    foods: list[dict[str, Any]],
) -> list[str]:
    """
    คืนลำดับธาตุของอาหารตามลำดับที่ระบบแสดงผล
    """
    return [
        food["recommended_element"]
        for food in foods
    ]


def display_category_result(
    category: str,
    foods: list[dict[str, Any]],
) -> None:
    """
    แสดงผลของแต่ละหมวดใน Terminal
    """

    counts = count_elements(foods)
    order = get_element_order(foods)

    print(f"\n[{category}]")

    for index, food in enumerate(
        foods,
        start=1,
    ):
        print(
            f"{index}. "
            f"{food['food_name_th']} "
            f"=> "
            f"{food['recommended_element']}"
        )

    print(
        "Count:",
        dict(counts),
    )

    print(
        "Order:",
        " → ".join(order),
    )


# =========================================================
# Generic Test Function
# =========================================================

def check_case(
    case_name: str,
    scores: dict[str, float],
    expected_mode: str,
    expected_primary: str,
    expected_secondary: str,
    expected_primary_count: int,
    expected_secondary_count: int,
) -> None:
    """
    ทดสอบ Recommendation 1 กรณี

    ตรวจ:
    1. mode
    2. primary / secondary
    3. quota
    4. จำนวนรายการ
    5. จำนวนอาหารแต่ละธาตุ
    6. ลำดับการแสดงผล
    """

    print()
    print("=" * 80)
    print(case_name)
    print("=" * 80)

    # -----------------------------------------------------
    # วิเคราะห์คะแนน
    # -----------------------------------------------------

    relationship = analyze_element_relationship(
        scores=scores,
        mixed_threshold=THRESHOLD,
    )

    print(
        "Scores:",
        scores,
    )

    print(
        "Mode:",
        relationship["mode"],
    )

    print(
        "Primary:",
        relationship["primary_element"],
    )

    print(
        "Secondary:",
        relationship["secondary_element"],
    )

    print(
        "Difference:",
        relationship["difference"],
    )

    # -----------------------------------------------------
    # ตรวจ Mode
    # -----------------------------------------------------

    assert (
        relationship["mode"]
        == expected_mode
    ), (
        f"Mode ผิด: "
        f"expected={expected_mode}, "
        f"actual={relationship['mode']}"
    )

    # -----------------------------------------------------
    # ตรวจ Primary
    # -----------------------------------------------------

    assert (
        relationship["primary_element"]
        == expected_primary
    ), (
        f"Primary ผิด: "
        f"expected={expected_primary}, "
        f"actual={relationship['primary_element']}"
    )

    # -----------------------------------------------------
    # ตรวจ Secondary
    # -----------------------------------------------------

    assert (
        relationship["secondary_element"]
        == expected_secondary
    ), (
        f"Secondary ผิด: "
        f"expected={expected_secondary}, "
        f"actual={relationship['secondary_element']}"
    )

    # -----------------------------------------------------
    # คำนวณ Quota
    # -----------------------------------------------------

    quotas = calculate_element_quotas(
        scores=scores,
        total_items=PER_CATEGORY,
        mixed_threshold=THRESHOLD,
    )

    expected_quota = {
        expected_primary:
            expected_primary_count,
        expected_secondary:
            expected_secondary_count,
    }

    print(
        "Expected quota:",
        expected_quota,
    )

    print(
        "Calculated quota:",
        quotas,
    )

    # -----------------------------------------------------
    # ตรวจ Primary Quota
    # -----------------------------------------------------

    assert (
        quotas[expected_primary]
        == expected_primary_count
    ), (
        f"Primary quota ผิด: "
        f"expected={expected_primary_count}, "
        f"actual={quotas[expected_primary]}"
    )

    # -----------------------------------------------------
    # ตรวจ Secondary Quota
    # -----------------------------------------------------

    assert (
        quotas[expected_secondary]
        == expected_secondary_count
    ), (
        f"Secondary quota ผิด: "
        f"expected={expected_secondary_count}, "
        f"actual={quotas[expected_secondary]}"
    )

    # -----------------------------------------------------
    # สร้าง Expected Order
    # -----------------------------------------------------

    expected_order = (
        [expected_primary]
        * expected_primary_count
        +
        [expected_secondary]
        * expected_secondary_count
    )

    print(
        "Expected order:",
        " → ".join(expected_order),
    )

    # -----------------------------------------------------
    # เรียก Recommendation จริง
    # -----------------------------------------------------

    grouped = recommend_foods_by_category(
        scores=scores,
        per_category=PER_CATEGORY,
        mixed_threshold=THRESHOLD,
    )

    # -----------------------------------------------------
    # ตรวจทุก Category
    # -----------------------------------------------------

    for category in CATEGORIES:

        foods = grouped[category]

        display_category_result(
            category,
            foods,
        )

        counts = count_elements(
            foods
        )

        actual_order = get_element_order(
            foods
        )

        # -------------------------------------------------
        # ต้องมี 4 รายการ
        # -------------------------------------------------

        assert (
            len(foods)
            == PER_CATEGORY
        ), (
            f"{category}: "
            f"ควรมี {PER_CATEGORY} รายการ "
            f"แต่ได้ {len(foods)}"
        )

        # -------------------------------------------------
        # ตรวจจำนวน Primary
        # -------------------------------------------------

        assert (
            counts[expected_primary]
            == expected_primary_count
        ), (
            f"{category}: "
            f"จำนวน {expected_primary} ผิด "
            f"expected={expected_primary_count}, "
            f"actual={counts[expected_primary]}"
        )

        # -------------------------------------------------
        # ตรวจจำนวน Secondary
        # -------------------------------------------------

        assert (
            counts[expected_secondary]
            == expected_secondary_count
        ), (
            f"{category}: "
            f"จำนวน {expected_secondary} ผิด "
            f"expected={expected_secondary_count}, "
            f"actual={counts[expected_secondary]}"
        )

        # -------------------------------------------------
        # ตรวจว่าไม่มีธาตุอื่นปะปน
        # -------------------------------------------------

        allowed_elements = {
            expected_primary,
            expected_secondary,
        }

        for element in counts:

            assert (
                element
                in allowed_elements
            ), (
                f"{category}: "
                f"พบธาตุที่ไม่ควรมี "
                f"{element}"
            )

        # -------------------------------------------------
        # ตรวจลำดับให้ตรงกับ UI
        # -------------------------------------------------

        assert (
            actual_order
            == expected_order
        ), (
            f"{category}: "
            "ลำดับการแสดงผลผิด\n"
            f"Expected: {expected_order}\n"
            f"Actual:   {actual_order}"
        )

    print()
    print(
        f"✅ {case_name} ผ่าน"
    )


# =========================================================
# CASE 1
# Equal → 2 + 2
#
# Expected UI:
# earth
# earth
# water
# water
# =========================================================

def test_equal_2_plus_2() -> None:

    scores = {
        "earth": 10.0,
        "water": 10.0,
        "wind": 5.0,
        "fire": 4.0,
    }

    check_case(
        case_name=(
            "CASE 1: EQUAL → 2+2"
        ),
        scores=scores,
        expected_mode="equal",
        expected_primary="earth",
        expected_secondary="water",
        expected_primary_count=2,
        expected_secondary_count=2,
    )


# =========================================================
# CASE 2
# Mixed → 3 + 1
#
# Expected UI:
# earth
# earth
# earth
# water
# =========================================================

def test_mixed_3_plus_1() -> None:

    scores = {
        "earth": 10.0,
        "water": 9.5,
        "wind": 5.0,
        "fire": 4.0,
    }

    check_case(
        case_name=(
            "CASE 2: MIXED → 3+1"
        ),
        scores=scores,
        expected_mode="mixed",
        expected_primary="earth",
        expected_secondary="water",
        expected_primary_count=3,
        expected_secondary_count=1,
    )


# =========================================================
# CASE 3
# Primary Only → 4 + 0
#
# Expected UI:
# earth
# earth
# earth
# earth
# =========================================================

def test_primary_only_4_plus_0() -> None:

    scores = {
        "earth": 10.0,
        "water": 8.0,
        "wind": 5.0,
        "fire": 4.0,
    }

    check_case(
        case_name=(
            "CASE 3: PRIMARY ONLY → 4+0"
        ),
        scores=scores,
        expected_mode="primary_only",
        expected_primary="earth",
        expected_secondary="water",
        expected_primary_count=4,
        expected_secondary_count=0,
    )


# =========================================================
# Boundary Test
# Difference = 1.0
#
# ต้องยังเป็น mixed
# =========================================================

def test_threshold_exactly_one() -> None:

    print()
    print("=" * 80)
    print(
        "BOUNDARY TEST: DIFFERENCE = 1.0"
    )
    print("=" * 80)

    scores = {
        "earth": 10.0,
        "water": 9.0,
        "wind": 5.0,
        "fire": 4.0,
    }

    relationship = analyze_element_relationship(
        scores=scores,
        mixed_threshold=THRESHOLD,
    )

    assert (
        relationship["mode"]
        == "mixed"
    ), (
        "ผลต่าง 1.0 ต้องเป็น mixed"
    )

    quotas = calculate_element_quotas(
        scores=scores,
        total_items=4,
        mixed_threshold=THRESHOLD,
    )

    assert (
        quotas["earth"] == 3
    )

    assert (
        quotas["water"] == 1
    )

    print(
        "Difference:",
        relationship["difference"],
    )

    print(
        "Mode:",
        relationship["mode"],
    )

    print(
        "Quota:",
        quotas,
    )

    print(
        "✅ Difference = 1.0 "
        "→ mixed 3+1 ผ่าน"
    )


# =========================================================
# Boundary Test
# Difference > 1.0
#
# ต้องเป็น primary_only
# =========================================================

def test_threshold_over_one() -> None:

    print()
    print("=" * 80)
    print(
        "BOUNDARY TEST: DIFFERENCE > 1.0"
    )
    print("=" * 80)

    scores = {
        "earth": 10.0,
        "water": 8.9,
        "wind": 5.0,
        "fire": 4.0,
    }

    relationship = analyze_element_relationship(
        scores=scores,
        mixed_threshold=THRESHOLD,
    )

    assert (
        relationship["mode"]
        == "primary_only"
    ), (
        "ผลต่างมากกว่า 1.0 "
        "ต้องเป็น primary_only"
    )

    quotas = calculate_element_quotas(
        scores=scores,
        total_items=4,
        mixed_threshold=THRESHOLD,
    )

    assert (
        quotas["earth"] == 4
    )

    assert (
        quotas["water"] == 0
    )

    print(
        "Difference:",
        relationship["difference"],
    )

    print(
        "Mode:",
        relationship["mode"],
    )

    print(
        "Quota:",
        quotas,
    )

    print(
        "✅ Difference > 1.0 "
        "→ primary_only 4+0 ผ่าน"
    )


# =========================================================
# Run All Tests
# =========================================================

def run_all_tests() -> None:

    print()
    print(
        "เริ่มทดสอบ Recommendation Quota"
    )

    print(
        f"Threshold = {THRESHOLD}"
    )

    print(
        f"Items per category = {PER_CATEGORY}"
    )

    # 2 + 2
    test_equal_2_plus_2()

    # 3 + 1
    test_mixed_3_plus_1()

    # 4 + 0
    test_primary_only_4_plus_0()

    # Boundary = 1.0
    test_threshold_exactly_one()

    # Boundary > 1.0
    test_threshold_over_one()

    print()
    print("=" * 80)

    print(
        "🎉 Recommendation Quota Tests "
        "ผ่านทั้งหมด"
    )

    print("=" * 80)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    run_all_tests()