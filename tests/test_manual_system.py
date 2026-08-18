from recommendation import (
    analyze_element_relationship,
    recommend_foods_by_category,
    get_avoid_rules,
    calculate_element_quotas,
)


# =========================================================
# Helper
# =========================================================

def print_result(title, scores):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("คะแนน:")
    for element, score in scores.items():
        print(f"- {element}: {score}")


    relationship = analyze_element_relationship(
        scores
    )

    print("\nผลวิเคราะห์:")
    print(
        "Mode:",
        relationship["mode"]
    )

    print(
        "ธาตุหลัก:",
        relationship["primary_element"]
    )

    print(
        "ธาตุรอง:",
        relationship["secondary_element"]
    )

    print(
        "ผลต่าง:",
        relationship["difference"]
    )


    quotas = calculate_element_quotas(
        scores,
        total_items=4
    )

    print("\nQuota (4 รายการ):")

    for element, amount in quotas.items():
        if amount > 0:
            print(
                f"- {element}: {amount}"
            )


    print("\nตัวอย่างอาหารแต่ละหมวด:")

    results = recommend_foods_by_category(
        scores,
        per_category=4
    )


    for category, foods in results.items():

        print("\n", category)

        for food in foods:

            print(
                "-",
                food["food_name_th"],
                "(",
                food["recommended_element_th"],
                ")"
            )


    print("\nอาหารที่ควรระวัง:")

    avoid = get_avoid_rules(
        scores
    )

    for item in avoid:
        print(
            "-",
            item["food_name_th"],
            "(",
            item["recommended_element_th"],
            ")"
        )



# =========================================================
# Test Case 1
# Equal
# คะแนนเท่ากัน
# Expected:
# mode = equal
# หลัก 2 + รอง 2
# =========================================================

equal_case = {

    "earth": 15,

    "water": 15,

    "wind": 10,

    "fire": 8,

}


print_result(
    "TEST CASE 1 : EQUAL",
    equal_case
)



# =========================================================
# Test Case 2
# Mixed
# ต่างกัน <= 1 คะแนน
# Expected:
# mode = mixed
# หลัก 3 + รอง 1
# =========================================================

mixed_case = {

    "earth": 16,

    "water": 15.5,

    "wind": 10,

    "fire": 8,

}


print_result(
    "TEST CASE 2 : MIXED",
    mixed_case
)



# =========================================================
# Test Case 3
# Primary Only
# ต่างกัน > 1 คะแนน
# Expected:
# mode = primary_only
# หลัก 4
# =========================================================

primary_only_case = {

    "earth": 18,

    "water": 15,

    "wind": 10,

    "fire": 8,

}


print_result(
    "TEST CASE 3 : PRIMARY ONLY",
    primary_only_case
)



print("\n")
print("=" * 70)
print("MANUAL SYSTEM TEST COMPLETE")
print("=" * 70)