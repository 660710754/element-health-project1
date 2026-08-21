from __future__ import annotations

from rag import (
    RAGSystem,
    build_rag_prompt,
    determine_active_elements,
    determine_mode,
    extract_sources,
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = (
    "scb10x/typhoon2.5-qwen3-4b:latest"
)

TOP_K = 5


# =========================================================
# TEST 1
# Build RAG Prompt
# =========================================================

def test_build_prompt() -> None:

    print()
    print("=" * 70)
    print("TEST 1 : Build RAG Prompt")
    print("=" * 70)

    prompt = build_rag_prompt(
        question=(
            "ธาตุไฟมีลักษณะอย่างไร"
        ),
        context=(
            "ธาตุไฟสัมพันธ์กับความร้อน "
            "และการย่อยอาหาร"
        ),
        active_elements=[
            "fire"
        ],
        primary_element=None,
        secondary_element=None,
    )

    assert (
        "ธาตุไฟมีลักษณะอย่างไร"
        in prompt
    )

    assert (
        "ความร้อน"
        in prompt
    )

    assert (
        "ธาตุไฟ"
        in prompt
    )

    print(
        "✓ สร้าง RAG Prompt ผ่าน"
    )


# =========================================================
# TEST 2
# Extract Sources
# =========================================================

def test_extract_sources() -> None:

    print()
    print("=" * 70)
    print("TEST 2 : Extract Sources")
    print("=" * 70)

    results = [

        {
            "source_file":
                "fire_profile.txt"
        },

        {
            "source_file":
                "fire_profile.txt"
        },

        {
            "source_file":
                "fire_food.txt"
        },

    ]

    sources = extract_sources(
        results
    )

    assert sources == [
        "fire_profile.txt",
        "fire_food.txt",
    ]

    print(
        "✓ ดึง Sources ไม่ซ้ำผ่าน"
    )


# =========================================================
# TEST 3
# Determine Active Element
# =========================================================

def test_determine_active_elements() -> None:

    print()
    print("=" * 70)
    print("TEST 3 : Determine Active Elements")
    print("=" * 70)

    # -----------------------------------------
    # Explicit element จากคำถาม
    # ต้องมี priority สูงกว่าผลแบบประเมิน
    # -----------------------------------------

    active = determine_active_elements(
        question=(
            "ธาตุไฟควรหลีกเลี่ยงอะไร"
        ),
        primary_element="water",
        secondary_element="earth",
    )

    assert active == [
        "fire"
    ]

    # -----------------------------------------
    # ไม่มี explicit element
    # ใช้ผลแบบประเมิน
    # -----------------------------------------

    active = determine_active_elements(
        question=(
            "ฉันควรกินอาหารอะไร"
        ),
        primary_element="water",
        secondary_element="earth",
    )

    assert active == [
        "water",
        "earth",
    ]

    print(
        "✓ Determine Active Elements ผ่าน"
    )


# =========================================================
# TEST 4
# Determine Mode
# =========================================================

def test_determine_mode() -> None:

    print()
    print("=" * 70)
    print("TEST 4 : Determine Mode")
    print("=" * 70)

    assert (
        determine_mode(
            ["fire"]
        )
        == "single"
    )

    assert (
        determine_mode(
            [
                "water",
                "earth",
            ]
        )
        == "mixed"
    )

    assert (
        determine_mode(
            []
        )
        == "unknown"
    )

    print(
        "✓ Determine Mode ผ่าน"
    )


# =========================================================
# TEST 5
# Retrieval Inside RAG
# =========================================================

def test_retrieval_inside_rag() -> None:

    print()
    print("=" * 70)
    print("TEST 5 : Retrieval Inside RAG")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    results = rag.retrieve(
        question=(
            "ธาตุลมมีลักษณะอย่างไร"
        ),
        active_elements=[
            "wind"
        ],
    )

    assert results

    assert len(
        results
    ) <= TOP_K

    assert any(
        (
            result["element"]
            == "wind"
        )
        for result in results
    )

    assert any(
        (
            result[
                "knowledge_type"
            ]
            == "profile"
        )
        for result in results
    )

    print(
        "✓ RAG เรียก Knowledge Retrieval ผ่าน"
    )


# =========================================================
# TEST 6
# Food Retrieval Inside RAG
# =========================================================

def test_food_retrieval_inside_rag() -> None:

    print()
    print("=" * 70)
    print("TEST 6 : Food Retrieval Inside RAG")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    results = rag.retrieve(
        question=(
            "ธาตุไฟควรกินอาหารอะไร"
        ),
        active_elements=[
            "fire"
        ],
    )

    assert results

    assert any(
        (
            item["element"]
            == "fire"
            and
            item[
                "knowledge_type"
            ]
            == "food"
        )
        for item in results
    )

    print(
        "✓ Food Retrieval ภายใน RAG ผ่าน"
    )


# =========================================================
# TEST 7
# Menu Retrieval Inside RAG
# =========================================================

def test_menu_retrieval_inside_rag() -> None:

    print()
    print("=" * 70)
    print("TEST 7 : Menu Retrieval Inside RAG")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    results = rag.retrieve(
        question=(
            "ธาตุลมมีเมนูอะไรแนะนำ"
        ),
        active_elements=[
            "wind"
        ],
    )

    assert results

    assert any(
        (
            item["element"]
            == "wind"
            and
            item[
                "knowledge_type"
            ]
            == "menu"
        )
        for item in results
    )

    print(
        "✓ Menu Retrieval ภายใน RAG ผ่าน"
    )


# =========================================================
# TEST 8
# Warning Retrieval Inside RAG
# =========================================================

def test_warning_retrieval_inside_rag() -> None:

    print()
    print("=" * 70)
    print("TEST 8 : Warning Retrieval Inside RAG")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    results = rag.retrieve(
        question=(
            "ธาตุน้ำควรหลีกเลี่ยงอาหารอะไร"
        ),
        active_elements=[
            "water"
        ],
    )

    assert results

    assert any(
        (
            item["element"]
            == "water"
            and
            item[
                "knowledge_type"
            ]
            == "warning"
        )
        for item in results
    )

    print(
        "✓ Warning Retrieval ภายใน RAG ผ่าน"
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

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    results = rag.retrieve(
        question=(
            "ฉันควรกินอาหารอะไร"
        ),
        active_elements=[
            "water",
            "earth",
        ],
    )

    assert results

    retrieved_elements = {
        item["element"]
        for item in results
    }

    assert (
        "water"
        in retrieved_elements
        or
        "earth"
        in retrieved_elements
    )

    assert any(
        (
            item[
                "knowledge_type"
            ]
            == "food"
        )
        for item in results
    )

    print(
        "✓ Mixed Element Retrieval ผ่าน"
    )


# =========================================================
# TEST 10
# Empty Question
# =========================================================

def test_empty_question() -> None:

    print()
    print("=" * 70)
    print("TEST 10 : Empty Question")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    result = rag.answer(
        question="   "
    )

    assert (
        result["sources"]
        == []
    )

    assert (
        result[
            "retrieved_chunks"
        ]
        == []
    )

    assert (
        result["answer"]
        == "กรุณาระบุคำถามก่อนใช้งานระบบ"
    )

    print(
        "✓ จัดการคำถามว่างผ่าน"
    )


# =========================================================
# TEST 11
# Full RAG Answer - Profile
#
# TEST นี้เรียก LLM จริง
# จึงอาจใช้เวลานาน
# =========================================================

def test_rag_profile_answer() -> None:

    print()
    print("=" * 70)
    print("TEST 11 : Full RAG Profile Answer")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    result = rag.answer(
        question=(
            "ธาตุน้ำมีลักษณะอย่างไร"
        )
    )

    assert (
        result["question"]
    )

    assert (
        result["answer"]
    )

    assert (
        result[
            "active_elements"
        ]
        == ["water"]
    )

    assert (
        result[
            "retrieved_chunks"
        ]
    )

    assert (
        "water_profile.txt"
        in result["sources"]
    )

    print(
        "✓ Full RAG Profile Answer ผ่าน"
    )


# =========================================================
# TEST 12
# Full RAG Answer - Food
#
# เรียก LLM จริง
# =========================================================

def test_rag_food_answer() -> None:

    print()
    print("=" * 70)
    print("TEST 12 : Full RAG Food Answer")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    result = rag.answer(
        question=(
            "ธาตุไฟควรกินอาหารอะไร"
        )
    )

    assert (
        result["answer"]
    )

    assert (
        result[
            "active_elements"
        ]
        == ["fire"]
    )

    assert (
        result[
            "retrieved_chunks"
        ]
    )

    assert any(
        (
            item[
                "knowledge_type"
            ]
            == "food"
        )
        for item
        in result[
            "retrieved_chunks"
        ]
    )

    print(
        "✓ Full RAG Food Answer ผ่าน"
    )


# =========================================================
# TEST 13
# Full RAG Answer - Assessment Mixed
#
# ตรวจว่าไม่มีชื่อธาตุในคำถาม
# แล้วระบบใช้ผลแบบประเมิน
# =========================================================

def test_rag_assessment_mixed_answer() -> None:

    print()
    print("=" * 70)
    print("TEST 13 : Full RAG Mixed Assessment")
    print("=" * 70)

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )

    result = rag.answer(
        question=(
            "ฉันควรกินอาหารอะไร"
        ),
        primary_element="water",
        secondary_element="earth",
    )

    assert (
        result["answer"]
    )

    assert (
        result[
            "active_elements"
        ]
        == [
            "water",
            "earth",
        ]
    )

    assert (
        result[
            "mode"
        ]
        == "mixed"
    )

    assert (
        result[
            "retrieved_chunks"
        ]
    )

    print(
        "✓ Full RAG Mixed Assessment ผ่าน"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "ทดสอบระบบ Retrieval-Augmented Generation"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Fast Tests
    # ไม่เรียก LLM
    # -----------------------------------------------------

    test_build_prompt()

    test_extract_sources()

    test_determine_active_elements()

    test_determine_mode()

    test_retrieval_inside_rag()

    test_food_retrieval_inside_rag()

    test_menu_retrieval_inside_rag()

    test_warning_retrieval_inside_rag()

    test_mixed_element_retrieval()

    test_empty_question()


    # -----------------------------------------------------
    # Full RAG Tests
    # เรียก Ollama จริง
    #
    # Codespaces อาจใช้เวลานาน
    # -----------------------------------------------------

    test_rag_profile_answer()

    test_rag_food_answer()

    test_rag_assessment_mixed_answer()


    print()
    print("=" * 70)

    print(
        "🎉 ทดสอบระบบ RAG สำเร็จทั้งหมด"
    )

    print("=" * 70)