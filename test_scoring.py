from scoring import calculate_scores, get_birth_element, get_result


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


def test_wind_score():
    answers = {
        "Q026": 2,
        "Q027": 2,
        "Q030": 2,
        "Q032": 2,
        "Q034": 2,
        "Q036": 2,
    }

    scores = calculate_scores(
        answers=answers,
        birth_month=5,
    )

    result = get_result(scores)

    assert result["primary_element"] == "wind"
    assert scores["wind"] > scores["earth"]
    assert scores["wind"] > scores["water"]
    assert scores["wind"] > scores["fire"]

    print("✓ ทดสอบกรณีธาตุลมผ่าน")


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
        answers=answers,
        birth_month=2,
    )

    result = get_result(scores)

    assert result["primary_element"] == "fire"

    print("✓ ทดสอบกรณีธาตุไฟผ่าน")


def test_invalid_month():
    try:
        get_birth_element(13)
    except ValueError:
        print("✓ ทดสอบเดือนเกิดผิดผ่าน")
    else:
        raise AssertionError("ระบบควรปฏิเสธเดือนเกิด 13")


def test_invalid_answer():
    try:
        calculate_scores(
            answers={"Q026": 5},
            birth_month=5,
        )
    except ValueError:
        print("✓ ทดสอบคะแนนผิดผ่าน")
    else:
        raise AssertionError("ระบบควรปฏิเสธคะแนนที่ไม่ใช่ 0, 1 หรือ 2")


if __name__ == "__main__":
    test_birth_element()
    test_wind_score()
    test_fire_score()
    test_invalid_month()
    test_invalid_answer()

    print()
    print("ทดสอบ scoring system สำเร็จทั้งหมด")