from __future__ import annotations

from food_retrieval import (
    retrieve_foods_by_element,
)


# =========================================================
# Helper
# =========================================================

def get_elements_from_results(
    results: list[dict],
) -> set[str]:

    return {
        item["recommended_element"]
        for item in results
    }



def print_results(
    title: str,
    results: list[dict],
) -> None:

    print("=" * 70)
    print(title)
    print("=" * 70)

    if not results:
        print("ไม่พบผลลัพธ์")
        return


    for index, food in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"{food['food_name_th']} "
            f"| {food['recommended_element_th']} "
            f"| score={food['ir_score']}"
        )



# =========================================================
# Test 1 : Primary Only
# =========================================================

def test_primary_only():

    """
    กรณีธาตุไฟเด่นเพียงธาตุเดียว

    Expected:
    - ผลลัพธ์ต้องมีเฉพาะ fire
    """

    results = retrieve_foods_by_element(
        active_elements=[
            "fire"
        ],
        top_k=5
    )


    elements = get_elements_from_results(
        results
    )


    assert elements == {
        "fire"
    }, (
        "Primary only ผิด: "
        f"พบธาตุ {elements}"
    )


    print_results(
        "TEST 1 : PRIMARY ONLY (fire)",
        results
    )


    print(
        "✅ Primary only ผ่าน\n"
    )



# =========================================================
# Test 2 : Mixed
# =========================================================

def test_mixed():

    """
    กรณีธาตุไฟ + ธาตุน้ำ

    Expected:
    - ต้องพบทั้ง fire และ water
    """

    results = retrieve_foods_by_element(
        active_elements=[
            "fire",
            "water"
        ],
        top_k=10
    )


    elements = get_elements_from_results(
        results
    )


    assert "fire" in elements, (
        "Mixed ไม่มีธาตุไฟ"
    )


    assert "water" in elements, (
        "Mixed ไม่มีธาตุน้ำ"
    )


    print_results(
        "TEST 2 : MIXED (fire + water)",
        results
    )


    print(
        "✅ Mixed ผ่าน\n"
    )



# =========================================================
# Test 3 : Equal
# =========================================================

def test_equal():

    """
    กรณีคะแนนเท่ากัน

    ระบบจะส่ง active elements
    เท่ากัน 2 ธาตุ

    Expected:
    - พบทั้งสองธาตุ
    """

    results = retrieve_foods_by_element(
        active_elements=[
            "earth",
            "water"
        ],
        top_k=10
    )


    elements = get_elements_from_results(
        results
    )


    assert "earth" in elements, (
        "Equal ไม่มีธาตุดิน"
    )


    assert "water" in elements, (
        "Equal ไม่มีธาตุน้ำ"
    )


    print_results(
        "TEST 3 : EQUAL (earth + water)",
        results
    )


    print(
        "✅ Equal ผ่าน\n"
    )



# =========================================================
# Test Data Validation
# =========================================================

def test_only_recommended_food():

    """
    ตรวจว่า IR ไม่คืนอาหาร avoid
    """

    results = retrieve_foods_by_element(
        active_elements=[
            "fire"
        ],
        top_k=10
    )


    for food in results:

        assert (
            food["recommendation_status"]
            ==
            "recommended"
        ), (
            "พบอาหาร avoid: "
            f"{food['food_name_th']}"
        )


    print(
        "✅ Recommendation status ผ่าน\n"
    )



# =========================================================
# Run All Tests
# =========================================================

def run_all_tests():

    print(
        "\nเริ่มทดสอบ Food Retrieval IR System\n"
    )


    test_primary_only()

    test_mixed()

    test_equal()

    test_only_recommended_food()


    print(
        "=" * 70
    )

    print(
        "🎉 Food Retrieval IR Test ผ่านทั้งหมด"
    )

    print(
        "=" * 70
    )



if __name__ == "__main__":

    run_all_tests()