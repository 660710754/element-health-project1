from __future__ import annotations

from intent_classifier import classify_question


# =========================================================
# Test Case สำหรับ Intent Classification
# =========================================================

TEST_CASES = [

    # -----------------------------------------
    # 1. Element Profile
    # -----------------------------------------

    {
        "question": "ธาตุไฟคืออะไร",
        "expected_intent": "element_info",
        "expected_element": "fire",
    },

    {
        "question": "คนธาตุดินมีลักษณะอย่างไร",
        "expected_intent": "profile",
        "expected_element": "earth",
    },

    {
        "question": "ธาตุน้ำมีลักษณะอย่างไร",
        "expected_intent": "profile",
        "expected_element": "water",
    },

    {
        "question": "ธาตุลมเป็นอย่างไร",
        "expected_intent": "profile",
        "expected_element": "wind",
    },


    # -----------------------------------------
    # 2. Food
    # -----------------------------------------

    {
        "question": "ธาตุไฟควรกินอะไร",
        "expected_intent": "food",
        "expected_element": "fire",
    },

    {
        "question": "คนธาตุน้ำควรกินอาหารอะไร",
        "expected_intent": "food",
        "expected_element": "water",
    },


    # -----------------------------------------
    # 3. Vegetable
    # -----------------------------------------

    {
        "question": "ธาตุไฟกินผักอะไร",
        "expected_intent": "food",
        "expected_element": "fire",
    },

    {
        "question": "ธาตุลมควรกินผักอะไร",
        "expected_intent": "food",
        "expected_element": "wind",
    },


    # -----------------------------------------
    # 4. Menu
    # -----------------------------------------

    {
        "question": "ธาตุลมมีเมนูอะไรบ้าง",
        "expected_intent": "menu",
        "expected_element": "wind",
    },

    {
        "question": "แนะนำเมนูอาหารสำหรับธาตุดิน",
        "expected_intent": "menu",
        "expected_element": "earth",
    },


    # -----------------------------------------
    # 5. Warning
    # -----------------------------------------

    {
        "question": "ธาตุน้ำควรหลีกเลี่ยงอาหารอะไร",
        "expected_intent": "warning",
        "expected_element": "water",
    },

    {
        "question": "คนธาตุไฟไม่ควรกินอะไร",
        "expected_intent": "warning",
        "expected_element": "fire",
    },


    # -----------------------------------------
    # 6. Symptom
    # -----------------------------------------

    {
        "question": (
            "ท้องอืด มีลม เวียนหัว "
            "เกี่ยวข้องกับธาตุอะไร"
        ),
        "expected_intent": "symptom",
        "expected_element": None,
    },


    # -----------------------------------------
    # 7. General
    # -----------------------------------------

    {
        "question": "ธาตุเจ้าเรือนปัจจุบันคืออะไร",
        "expected_intent": "element_info",
        "expected_element": None,
    },

]


# =========================================================
# Test Function
# =========================================================

def test_intent_classifier():

    passed = 0
    failed = 0


    print()
    print("=" * 70)
    print("ทดสอบ Intent Classifier")
    print("=" * 70)


    for case in TEST_CASES:

        result = classify_question(
            case["question"]
        )


        intent = result.get(
            "intent"
        )

        element = result.get(
            "element"
        )


        is_intent_correct = (
            intent
            ==
            case["expected_intent"]
        )


        is_element_correct = (
            element
            ==
            case["expected_element"]
        )


        if (
            is_intent_correct
            and
            is_element_correct
        ):

            print(
                "✓ ผ่าน |",
                case["question"]
            )

            passed += 1


        else:

            print(
                "✗ ไม่ผ่าน |",
                case["question"]
            )

            print(
                "   ได้:",
                result
            )

            print(
                "   ควรได้:",
                {
                    "intent":
                        case["expected_intent"],
                    "element":
                        case["expected_element"],
                }
            )

            failed += 1



    print()

    print("=" * 70)

    print(
        f"ผ่าน: {passed}"
    )

    print(
        f"ไม่ผ่าน: {failed}"
    )

    print("=" * 70)



    assert failed == 0, (
        "Intent Classifier มีบางกรณีไม่ผ่าน"
    )



# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    test_intent_classifier()

    print()

    print(
        "✓ ทดสอบ intent_classifier.py สำเร็จทั้งหมด"
    )