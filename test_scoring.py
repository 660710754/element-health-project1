from scoring import (
    calculate_scores,
    get_birth_element,
    get_result,
    classify_recommendation_case
)


# =========================================================
# Test Birth Element
# ทดสอบธาตุเจ้าเรือนจากเดือนเกิด
# =========================================================

def test_birth_element():

    assert get_birth_element(1) == "fire"
    assert get_birth_element(3) == "fire"

    assert get_birth_element(4) == "wind"
    assert get_birth_element(6) == "wind"

    assert get_birth_element(7) == "water"
    assert get_birth_element(9) == "water"

    assert get_birth_element(10) == "earth"
    assert get_birth_element(12) == "earth"

    print("✓ ทดสอบธาตุเกิดผ่าน")


# =========================================================
# Test Wind Score
# ทดสอบคะแนนธาตุลมจากแบบสอบถาม
# =========================================================

def test_wind_score():

    answers = {
        "Q026": 2,
        "Q027": 2,
        "Q030": 2,
        "Q032": 2,
        "Q034": 2,
        "Q036": 2,
    }

    # คำนวณจากแบบสอบถามเท่านั้น
    scores = calculate_scores(
        answers=answers
    )

    result = get_result(scores)

    assert result["primary_element"] == "wind"
    assert scores["wind"] > scores["earth"]
    assert scores["wind"] > scores["water"]
    assert scores["wind"] > scores["fire"]

    print("✓ ทดสอบกรณีธาตุลมผ่าน")


# =========================================================
# Test Fire Score
# ทดสอบคะแนนธาตุไฟจากแบบสอบถาม
# =========================================================

def test_fire_score():

    answers = {
        "Q038": 2,
        "Q039": 2,
        "Q041": 2,
        "Q042": 2,
        "Q048": 2,
        "Q049": 2,
    }

    scores = calculate_scores(
        answers=answers
    )

    result = get_result(scores)

    assert result["primary_element"] == "fire"

    print("✓ ทดสอบกรณีธาตุไฟผ่าน")


# =========================================================
# Test Invalid Month
# ทดสอบเดือนเกิดไม่ถูกต้อง
# =========================================================

def test_invalid_month():

    try:
        get_birth_element(13)

    except ValueError:
        print("✓ ทดสอบเดือนเกิดผิดผ่าน")

    else:
        raise AssertionError(
            "ระบบควรปฏิเสธเดือนเกิด 13"
        )


# =========================================================
# Test Invalid Answer
# ทดสอบคะแนนคำตอบไม่ถูกต้อง
# =========================================================

def test_invalid_answer():

    try:
        calculate_scores(
            answers={"Q026": 5}
        )

    except ValueError:
        print("✓ ทดสอบคะแนนผิดผ่าน")

    else:
        raise AssertionError(
            "ระบบควรปฏิเสธคะแนนที่ไม่ใช่ 0, 1 หรือ 2"
        )


# =========================================================
# Test Recommendation Case
# ทดสอบการจำแนก 3 กรณี
# =========================================================

def test_equal_case():

    result = classify_recommendation_case(
        10, 10
    )

    assert result["recommendation_case"] == "Equal"
    assert result["recommendation_ratio"] == (2, 2)

    print("✓ ทดสอบ Equal ผ่าน")


def test_mixed_case():

    result = classify_recommendation_case(
        10, 9.5
    )

    assert result["recommendation_case"] == "Mixed"
    assert result["recommendation_ratio"] == (3, 1)

    print("✓ ทดสอบ Mixed ผ่าน")


def test_primary_only_case():

    result = classify_recommendation_case(
        10, 8
    )

    assert result["recommendation_case"] == "Primary Only"
    assert result["recommendation_ratio"] == (4, 0)

    print("✓ ทดสอบ Primary Only ผ่าน")


# =========================================================
# Test Birth Month Separation
# ทดสอบว่าเดือนเกิดไม่เปลี่ยนคะแนนแบบสอบถาม
# =========================================================

def test_birth_month_separate_from_scores():

    answers = {
        "Q026": 2,
        "Q027": 2,
        "Q030": 2,
    }

    scores = calculate_scores(answers)

    result = get_result(
        scores,
        birth_month=5
    )

    assert result["birth_element"] == "wind"

    # คะแนนต้องมาจากแบบสอบถามเท่านั้น
    assert scores == calculate_scores(answers)

    print("✓ ทดสอบการแยกเดือนเกิดจากคะแนนผ่าน")


# =========================================================
# Run Tests
# =========================================================

if __name__ == "__main__":

    test_birth_element()
    test_wind_score()
    test_fire_score()
    test_invalid_month()
    test_invalid_answer()

    test_equal_case()
    test_mixed_case()
    test_primary_only_case()

    test_birth_month_separate_from_scores()

    print()
    print("ทดสอบ scoring system สำเร็จทั้งหมด")
