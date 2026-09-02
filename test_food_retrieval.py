from __future__ import annotations

from food_retrieval import (
    FoodRetriever,
    detect_category_intent,
    detect_taste_intents,
    retrieve_foods,
    retrieve_foods_by_element,
)


# =========================================================
# Helper
# =========================================================

def get_elements_from_results(
    results: list[dict],
) -> set[str]:
    """
    ดึงชื่อธาตุจากผลลัพธ์
    """

    return {
        item["recommended_element"]
        for item in results
    }


def print_results(
    title: str,
    results: list[dict],
) -> None:
    """
    แสดงผลลัพธ์สำหรับตรวจสอบใน Terminal
    """

    print("=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("ไม่พบผลลัพธ์")
        print()
        return

    for index, food in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"{food['food_name_th']} "
            f"| element={food['recommended_element']} "
            f"| category={food.get('category', '')} "
            f"| taste={food.get('food_taste_profile', '')} "
            f"| score={food.get('ir_score', '')}"
        )

    print()


# =========================================================
# Test 1 : Load Database
# =========================================================

def test_load_database():
    """
    ตรวจว่า foods.csv โหลดได้
    และมี food_taste_profile
    """

    retriever = FoodRetriever()

    foods = retriever.all_foods

    assert foods, (
        "โหลด foods.csv ไม่สำเร็จ"
    )

    assert len(foods) > 0, (
        "foods.csv ไม่มีข้อมูล"
    )

    assert (
        "food_taste_profile"
        in foods[0]
    ), (
        "ไม่พบคอลัมน์ "
        "food_taste_profile "
        "ใน foods.csv"
    )

    print(
        f"✅ Load Database ผ่าน "
        f"({len(foods)} rows)\n"
    )


# =========================================================
# Test 2 : Primary Element Filtering
# =========================================================

def test_primary_only():
    """
    ทดสอบการค้นแบบธาตุเดียว

    Expected:
    - เมื่อ active_elements = fire
    - ผลลัพธ์ต้องเป็น fire เท่านั้น
    """

    results = (
        retrieve_foods_by_element(
            active_elements=["fire"],
            top_k=5
        )
    )

    assert results, (
        "Primary Only ไม่พบผลลัพธ์"
    )

    elements = (
        get_elements_from_results(
            results
        )
    )

    assert elements == {"fire"}, (
        "Primary Only ผิด "
        f"พบธาตุ {elements}"
    )

    print_results(
        "TEST 2 : PRIMARY ONLY (fire)",
        results
    )

    print(
        "✅ Primary Element Filtering ผ่าน\n"
    )


# =========================================================
# Test 3 : Multiple Element Filtering
# =========================================================

def test_multiple_elements():
    """
    ตรวจว่าเมื่อกำหนดหลายธาตุ
    IR จะไม่คืนอาหารนอกเหนือจากธาตุที่กำหนด

    หมายเหตุ:
    ไม่บังคับว่า Top-K ต้องมีครบทุกธาตุ
    เพราะการ Balance Primary/Secondary
    เป็นหน้าที่ของ recommendation.py
    """

    expected_elements = {
        "fire",
        "water"
    }

    results = (
        retrieve_foods_by_element(
            active_elements=[
                "fire",
                "water"
            ],
            top_k=10
        )
    )

    assert results, (
        "Multiple Elements "
        "ไม่พบผลลัพธ์"
    )

    found_elements = (
        get_elements_from_results(
            results
        )
    )

    assert (
        found_elements
        <=
        expected_elements
    ), (
        "พบอาหารจากธาตุที่ไม่ได้ร้องขอ: "
        f"{found_elements}"
    )

    print_results(
        "TEST 3 : MULTIPLE ELEMENTS "
        "(fire + water)",
        results
    )

    print(
        "✅ Multiple Element Filtering ผ่าน\n"
    )


# =========================================================
# Test 4 : Recommendation Status
# =========================================================

def test_only_recommended_food():
    """
    ตรวจว่า IR ไม่คืนรายการ avoid
    """

    results = (
        retrieve_foods_by_element(
            active_elements=["fire"],
            top_k=10
        )
    )

    assert results, (
        "ไม่พบผลลัพธ์สำหรับทดสอบ "
        "recommendation_status"
    )

    for food in results:

        assert (
            food[
                "recommendation_status"
            ]
            ==
            "recommended"
        ), (
            "พบอาหาร avoid: "
            f"{food['food_name_th']}"
        )

    print(
        "✅ Recommendation Status ผ่าน\n"
    )


# =========================================================
# Test 5 : Category Intent Detection
# =========================================================

def test_category_intent():
    """
    ตรวจการจับหมวดอาหารจาก User Input
    """

    test_cases = [
        (
            "ธาตุไฟควรกินผักอะไร",
            "vegetable_herb"
        ),
        (
            "ธาตุน้ำมีผลไม้อะไร",
            "fruit"
        ),
        (
            "ธาตุลมมีเมนูอะไร",
            "menu"
        ),
        (
            "ธาตุดินมีขนมอะไร",
            "snack"
        ),
        (
            "มีเครื่องดื่มอะไรแนะนำ",
            "drink"
        ),
    ]

    for query, expected in test_cases:

        detected = (
            detect_category_intent(
                query
            )
        )

        assert detected == expected, (
            f"Category Intent ผิด\n"
            f"Query: {query}\n"
            f"Expected: {expected}\n"
            f"Detected: {detected}"
        )

    print(
        "✅ Category Intent Detection ผ่าน\n"
    )


# =========================================================
# Test 6 : Taste Intent Detection
# =========================================================

def test_taste_intent():
    """
    ตรวจการจับรสชาติจาก User Input
    """

    test_cases = [
        (
            "อาหารรสขม",
            ["ขม"]
        ),
        (
            "ผลไม้รสเปรี้ยว",
            ["เปรี้ยว"]
        ),
        (
            "ขนมรสหวาน",
            ["หวาน"]
        ),
        (
            "เมนูรสเผ็ด",
            ["เผ็ด"]
        ),
    ]

    for query, expected in test_cases:

        detected = (
            detect_taste_intents(
                query
            )
        )

        assert detected == expected, (
            f"Taste Intent ผิด\n"
            f"Query: {query}\n"
            f"Expected: {expected}\n"
            f"Detected: {detected}"
        )

    print(
        "✅ Taste Intent Detection ผ่าน\n"
    )


# =========================================================
# Test 7 : Fire + Vegetable + Bitter
# =========================================================

def test_fire_bitter_vegetable():
    """
    User Query:
    ธาตุไฟควรกินผักรสขมอะไร

    Expected:
    - fire
    - vegetable_herb
    - food_taste_profile มี "ขม"
    """

    query = (
        "ธาตุไฟควรกินผักรสขมอะไร"
    )

    results = (
        retrieve_foods(
            query=query,
            active_elements=["fire"],
            top_k=5
        )
    )

    assert results, (
        "ไม่พบอาหารสำหรับ Query "
        "ธาตุไฟ + ผัก + ขม"
    )

    for food in results:

        assert (
            food[
                "recommended_element"
            ]
            ==
            "fire"
        ), (
            "พบอาหารผิดธาตุ: "
            f"{food['food_name_th']}"
        )

        assert (
            food[
                "category"
            ]
            ==
            "vegetable_herb"
        ), (
            "พบอาหารผิดหมวด: "
            f"{food['food_name_th']} "
            f"({food['category']})"
        )

        assert (
            "ขม"
            in food.get(
                "food_taste_profile",
                ""
            )
        ), (
            "พบอาหารที่ไม่ตรงรสขม: "
            f"{food['food_name_th']} "
            f"= "
            f"{food.get('food_taste_profile', '')}"
        )

    names = {
        food["food_name_th"]
        for food in results
    }

    assert "มะระ" in names, (
        "Query ผักรสขมของธาตุไฟ "
        "ควรพบมะระ"
    )

    print_results(
        "TEST 7 : FIRE + VEGETABLE + BITTER",
        results
    )

    print(
        "✅ Fire Bitter Vegetable ผ่าน\n"
    )


# =========================================================
# Test 8 : Water + Fruit + Sour
# =========================================================

def test_water_sour_fruit():
    """
    User Query:
    ธาตุน้ำมีผลไม้รสเปรี้ยวอะไร

    Expected:
    - water
    - fruit
    - food_taste_profile มี "เปรี้ยว"
    """

    query = (
        "ธาตุน้ำมีผลไม้รสเปรี้ยวอะไร"
    )

    results = (
        retrieve_foods(
            query=query,
            active_elements=["water"],
            top_k=5
        )
    )

    assert results, (
        "ไม่พบอาหารสำหรับ "
        "ธาตุน้ำ + ผลไม้ + เปรี้ยว"
    )

    for food in results:

        assert (
            food[
                "recommended_element"
            ]
            ==
            "water"
        )

        assert (
            food[
                "category"
            ]
            ==
            "fruit"
        ), (
            "พบรายการที่ไม่ใช่ผลไม้: "
            f"{food['food_name_th']}"
        )

        assert (
            "เปรี้ยว"
            in food.get(
                "food_taste_profile",
                ""
            )
        ), (
            "พบผลไม้ที่ไม่ตรงรสเปรี้ยว: "
            f"{food['food_name_th']}"
        )

    print_results(
        "TEST 8 : WATER + FRUIT + SOUR",
        results
    )

    print(
        "✅ Water Sour Fruit ผ่าน\n"
    )


# =========================================================
# Test 9 : Wind + Menu + Spicy
# =========================================================

def test_wind_spicy_menu():
    """
    User Query:
    ธาตุลมมีเมนูรสเผ็ดอะไร

    Expected:
    - wind
    - menu
    - food_taste_profile มี "เผ็ด"
    """

    query = (
        "ธาตุลมมีเมนูรสเผ็ดอะไร"
    )

    results = (
        retrieve_foods(
            query=query,
            active_elements=["wind"],
            top_k=5
        )
    )

    assert results, (
        "ไม่พบอาหารสำหรับ "
        "ธาตุลม + เมนู + เผ็ด"
    )

    for food in results:

        assert (
            food[
                "recommended_element"
            ]
            ==
            "wind"
        )

        assert (
            food[
                "category"
            ]
            ==
            "menu"
        ), (
            "พบรายการที่ไม่ใช่ menu: "
            f"{food['food_name_th']}"
        )

        assert (
            "เผ็ด"
            in food.get(
                "food_taste_profile",
                ""
            )
        ), (
            "พบเมนูที่ไม่ตรงรสเผ็ด: "
            f"{food['food_name_th']}"
        )

    print_results(
        "TEST 9 : WIND + MENU + SPICY",
        results
    )

    print(
        "✅ Wind Spicy Menu ผ่าน\n"
    )


# =========================================================
# Test 10 : Earth + Snack + Sweet
# =========================================================

def test_earth_sweet_snack():
    """
    User Query:
    ธาตุดินมีขนมรสหวานอะไร

    Expected:
    - earth
    - snack
    - food_taste_profile มี "หวาน"
    """

    query = (
        "ธาตุดินมีขนมรสหวานอะไร"
    )

    results = (
        retrieve_foods(
            query=query,
            active_elements=["earth"],
            top_k=5
        )
    )

    assert results, (
        "ไม่พบอาหารสำหรับ "
        "ธาตุดิน + ขนม + หวาน"
    )

    for food in results:

        assert (
            food[
                "recommended_element"
            ]
            ==
            "earth"
        )

        assert (
            food[
                "category"
            ]
            ==
            "snack"
        ), (
            "พบรายการที่ไม่ใช่ snack: "
            f"{food['food_name_th']}"
        )

        assert (
            "หวาน"
            in food.get(
                "food_taste_profile",
                ""
            )
        ), (
            "พบขนมที่ไม่ตรงรสหวาน: "
            f"{food['food_name_th']}"
        )

    print_results(
        "TEST 10 : EARTH + SNACK + SWEET",
        results
    )

    print(
        "✅ Earth Sweet Snack ผ่าน\n"
    )


# =========================================================
# Test 11 : User Query Really Affects Retrieval
# =========================================================

def test_different_queries_change_results():
    """
    ตรวจว่าคำถามของ User
    มีผลต่อ Retrieval จริง

    ใช้ธาตุเดียวกัน แต่เปลี่ยน Intent
    """

    bitter_query = (
        "ธาตุไฟควรกินผักรสขมอะไร"
    )

    sweet_query = (
        "ธาตุไฟมีขนมรสหวานอะไร"
    )

    bitter_results = (
        retrieve_foods(
            query=bitter_query,
            active_elements=["fire"],
            top_k=5
        )
    )

    sweet_results = (
        retrieve_foods(
            query=sweet_query,
            active_elements=["fire"],
            top_k=5
        )
    )

    bitter_names = [
        item["food_name_th"]
        for item in bitter_results
    ]

    sweet_names = [
        item["food_name_th"]
        for item in sweet_results
    ]

    assert (
        bitter_names
        !=
        sweet_names
    ), (
        "เปลี่ยน User Query แล้ว "
        "ผล Retrieval ยังเหมือนเดิม"
    )

    print_results(
        "TEST 11A : FIRE BITTER VEGETABLE",
        bitter_results
    )

    print_results(
        "TEST 11B : FIRE SWEET SNACK",
        sweet_results
    )

    print(
        "✅ User Query มีผลต่อ Retrieval ผ่าน\n"
    )


# =========================================================
# Test 12 : Debug Intent Metadata
# =========================================================

def test_intent_metadata():
    """
    ตรวจว่า Result เก็บข้อมูล Intent
    เพื่อใช้ Debug/Evaluation ได้
    """

    results = (
        retrieve_foods(
            query=(
                "ธาตุไฟควรกินผักรสขมอะไร"
            ),
            active_elements=["fire"],
            top_k=3
        )
    )

    assert results

    for food in results:

        assert (
            food.get(
                "detected_category"
            )
            ==
            "vegetable_herb"
        )

        assert (
            "ขม"
            in food.get(
                "detected_tastes",
                ""
            )
        )

    print(
        "✅ Intent Metadata ผ่าน\n"
    )


# =========================================================
# Run All Tests
# =========================================================

def run_all_tests():

    print(
        "\n"
        "เริ่มทดสอบ Food Retrieval "
        "Hybrid IR System"
        "\n"
    )

    test_load_database()

    test_primary_only()

    test_multiple_elements()

    test_only_recommended_food()

    test_category_intent()

    test_taste_intent()

    test_fire_bitter_vegetable()

    test_water_sour_fruit()

    test_wind_spicy_menu()

    test_earth_sweet_snack()

    test_different_queries_change_results()

    test_intent_metadata()

    print(
        "=" * 80
    )

    print(
        "🎉 Food Retrieval IR Test "
        "ผ่านทั้งหมด"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    run_all_tests()