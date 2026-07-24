from recommendation import (
    build_recommendation_summary,
    get_avoid_rules,
    load_foods,
    normalize_scores,
    recommend_foods,
    recommend_foods_by_category,
)


def test_load_foods():
    foods = load_foods()

    assert len(foods) > 0
    assert "food_id" in foods[0]
    assert "food_name_th" in foods[0]

    print("✓ โหลด foods.csv ผ่าน")


def test_normalize_scores():
    scores = {
        "earth": 10,
        "water": 20,
        "wind": 30,
        "fire": 40,
    }

    normalized = normalize_scores(scores)

    assert round(sum(normalized.values()), 6) == 1.0
    assert normalized["fire"] == 0.4

    print("✓ Normalize คะแนนผ่าน")


def test_wind_recommendation():
    scores = {
        "earth": 2,
        "water": 3,
        "wind": 20,
        "fire": 1,
    }

    recommendations = recommend_foods(
        scores=scores,
        limit=10,
    )

    assert len(recommendations) == 10
    assert recommendations[0]["recommended_element"] == "wind"

    print("✓ แนะนำอาหารธาตุลมผ่าน")


def test_category_recommendation():
    scores = {
        "earth": 5,
        "water": 20,
        "wind": 3,
        "fire": 2,
    }

    grouped = recommend_foods_by_category(
        scores=scores,
        per_category=2,
    )

    assert "menu" in grouped
    assert "fruit" in grouped
    assert "drink" in grouped

    print("✓ แนะนำแยกหมวดผ่าน")


def test_avoid_rules():
    scores = {
        "earth": 1,
        "water": 2,
        "wind": 3,
        "fire": 20,
    }

    avoid_rules = get_avoid_rules(scores)

    assert len(avoid_rules) > 0
    assert any(
        item["recommended_element"] == "fire"
        for item in avoid_rules
    )

    print("✓ ข้อควรหลีกเลี่ยงผ่าน")


def test_summary():
    scores = {
        "earth": 10,
        "water": 11,
        "wind": 20,
        "fire": 5,
    }

    summary = build_recommendation_summary(scores)

    assert summary["primary_element"] == "wind"
    assert "top_recommendations" in summary
    assert "avoid_rules" in summary

    print("✓ สร้างสรุปผลผ่าน")


if __name__ == "__main__":
    test_load_foods()
    test_normalize_scores()
    test_wind_recommendation()
    test_category_recommendation()
    test_avoid_rules()
    test_summary()

    print()
    print("ทดสอบระบบแนะนำอาหารสำเร็จทั้งหมด")