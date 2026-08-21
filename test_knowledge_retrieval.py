from __future__ import annotations

from knowledge_retrieval import (
    load_element_knowledge,
    retrieve_knowledge,
    build_knowledge_context,
    detect_explicit_elements,
    detect_query_intents,
)


# =========================================================
# Test Configuration
# =========================================================

TEST_ELEMENTS = [
    "earth",
    "water",
    "wind",
    "fire",
]


REQUIRED_SECTIONS = [
    "profile",
    "food",
    "menu",
    "warning",
]


TOP_K = 5


# =========================================================
# Helper
# =========================================================

def assert_has_expected_result(
    results: list[dict],
    expected_element: str,
    expected_type: str,
) -> None:
    """
    ตรวจว่าใน Top-K มีอย่างน้อย 1 chunk
    ที่ตรงทั้ง element และ knowledge_type
    """

    assert results, (
        "ไม่พบผลลัพธ์จาก Knowledge Retrieval"
    )

    matched = any(
        (
            item["element"] == expected_element
            and item["knowledge_type"] == expected_type
        )
        for item in results
    )

    assert matched, (
        f"ไม่พบผลลัพธ์ที่ตรงกับ "
        f"element={expected_element}, "
        f"type={expected_type}"
    )


# =========================================================
# TEST 1
# Load Single Element
# =========================================================

def test_single_element_loading() -> None:

    print("=" * 70)
    print("TEST 1 : Load Single Element Knowledge")
    print("=" * 70)

    element = "earth"

    knowledge = load_element_knowledge(
        element
    )

    assert isinstance(
        knowledge,
        dict
    )

    for section in REQUIRED_SECTIONS:

        assert section in knowledge, (
            f"Missing section: {section}"
        )

        assert len(
            knowledge[section].strip()
        ) > 0, (
            f"{element}_{section}.txt is empty"
        )

    print(
        "✓ Earth knowledge loaded"
    )


# =========================================================
# TEST 2
# Load All Elements
# =========================================================

def test_all_elements_loading() -> None:

    print()
    print("=" * 70)
    print("TEST 2 : Load All Element Knowledge")
    print("=" * 70)

    for element in TEST_ELEMENTS:

        knowledge = load_element_knowledge(
            element
        )

        for section in REQUIRED_SECTIONS:

            assert section in knowledge

            assert knowledge[
                section
            ].strip() != ""

        print(
            f"✓ {element} knowledge loaded"
        )


# =========================================================
# TEST 3
# Explicit Element Detection
# =========================================================

def test_explicit_element_detection() -> None:

    print()
    print("=" * 70)
    print("TEST 3 : Explicit Element Detection")
    print("=" * 70)

    query = (
        "ธาตุน้ำและธาตุดินควรกินอะไร"
    )

    detected = detect_explicit_elements(
        query
    )

    assert "water" in detected
    assert "earth" in detected

    print(
        "✓ Explicit element detection passed"
    )


# =========================================================
# TEST 4
# Intent Detection
# =========================================================

def test_intent_detection() -> None:

    print()
    print("=" * 70)
    print("TEST 4 : Query Intent Detection")
    print("=" * 70)

    query = (
        "ธาตุน้ำควรกินอะไร "
        "และควรหลีกเลี่ยงอะไร"
    )

    intents = detect_query_intents(
        query
    )

    assert "food" in intents
    assert "warning" in intents

    print(
        "✓ Query intent detection passed"
    )


# =========================================================
# TEST 5
# Profile Retrieval
# =========================================================

def test_profile_retrieval() -> None:

    print()
    print("=" * 70)
    print("TEST 5 : Profile Retrieval")
    print("=" * 70)

    query = (
        "ธาตุน้ำมีลักษณะอย่างไร"
    )

    results = retrieve_knowledge(
        query=query,
        active_elements=[
            "water"
        ],
        top_k=TOP_K,
    )

    assert_has_expected_result(
        results=results,
        expected_element="water",
        expected_type="profile",
    )

    print(
        "✓ Profile retrieval passed"
    )


# =========================================================
# TEST 6
# Food Retrieval
# =========================================================

def test_food_retrieval() -> None:

    print()
    print("=" * 70)
    print("TEST 6 : Food Retrieval")
    print("=" * 70)

    query = (
        "ธาตุน้ำควรกินอาหารอะไร"
    )

    results = retrieve_knowledge(
        query=query,
        active_elements=[
            "water"
        ],
        top_k=TOP_K,
    )

    assert_has_expected_result(
        results=results,
        expected_element="water",
        expected_type="food",
    )

    print(
        "✓ Food retrieval passed"
    )


# =========================================================
# TEST 7
# Menu Retrieval
# =========================================================

def test_menu_retrieval() -> None:

    print()
    print("=" * 70)
    print("TEST 7 : Menu Retrieval")
    print("=" * 70)

    query = (
        "ธาตุลมมีเมนูอะไรแนะนำ"
    )

    results = retrieve_knowledge(
        query=query,
        active_elements=[
            "wind"
        ],
        top_k=TOP_K,
    )

    assert_has_expected_result(
        results=results,
        expected_element="wind",
        expected_type="menu",
    )

    print(
        "✓ Menu retrieval passed"
    )


# =========================================================
# TEST 8
# Warning Retrieval
# =========================================================

def test_warning_retrieval() -> None:

    print()
    print("=" * 70)
    print("TEST 8 : Warning Retrieval")
    print("=" * 70)

    query = (
        "ธาตุไฟควรหลีกเลี่ยงอะไร"
    )

    results = retrieve_knowledge(
        query=query,
        active_elements=[
            "fire"
        ],
        top_k=TOP_K,
    )

    assert_has_expected_result(
        results=results,
        expected_element="fire",
        expected_type="warning",
    )

    print(
        "✓ Warning retrieval passed"
    )


# =========================================================
# TEST 9
# Mixed Element Retrieval
# =========================================================

def test_mixed_element_retrieval() -> None:

    print()
    print("=" * 70)
    print("TEST 9 : Mixed Element Retrieval")
    print("=" * 70)

    query = (
        "จากผลธาตุน้ำและธาตุดิน "
        "ควรกินอาหารอะไร"
    )

    results = retrieve_knowledge(
        query=query,
        active_elements=[
            "water",
            "earth",
        ],
        top_k=TOP_K,
    )

    assert results

    retrieved_elements = {
        item["element"]
        for item in results
    }

    assert (
        "water" in retrieved_elements
        or "earth" in retrieved_elements
    )

    assert any(
        item[
            "knowledge_type"
        ] == "food"
        for item in results
    )

    print(
        "✓ Mixed element retrieval passed"
    )


# =========================================================
# TEST 10
# Build RAG Context
# =========================================================

def test_build_context() -> None:

    print()
    print("=" * 70)
    print("TEST 10 : Build RAG Context")
    print("=" * 70)

    context = build_knowledge_context(
        query=(
            "ธาตุไฟควรกินอาหารอะไร"
        ),
        active_elements=[
            "fire"
        ],
        top_k=TOP_K,
    )

    assert isinstance(
        context,
        str
    )

    assert len(
        context.strip()
    ) > 0

    assert (
        "ธาตุไฟ"
        in context
    )

    assert (
        "แหล่งข้อมูล:"
        in context
    )

    print(
        "✓ RAG context generation passed"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print(
        "\nเริ่มทดสอบ Knowledge Retrieval System\n"
    )

    test_single_element_loading()

    test_all_elements_loading()

    test_explicit_element_detection()

    test_intent_detection()

    test_profile_retrieval()

    test_food_retrieval()

    test_menu_retrieval()

    test_warning_retrieval()

    test_mixed_element_retrieval()

    test_build_context()

    print()
    print("=" * 70)

    print(
        "🎉 Knowledge Retrieval Test ผ่านทั้งหมด"
    )

    print("=" * 70)