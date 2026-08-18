from __future__ import annotations

from rag import (
    RAGSystem,
    build_rag_prompt,
    extract_sources,
)


MODEL_NAME = "scb10x/typhoon2.5-qwen3-4b:latest"


# =========================================================
# Test 1: ทดสอบสร้าง Prompt
# =========================================================

def test_build_prompt():

    prompt = build_rag_prompt(
        question="ธาตุไฟมีลักษณะอย่างไร",
        context=(
            "ธาตุไฟสัมพันธ์กับความร้อน "
            "และการย่อยอาหาร"
        ),
        classification={
            "intent": "profile",
            "element": "fire",
        },
    )

    assert (
        "ธาตุไฟมีลักษณะอย่างไร"
        in prompt
    )

    assert (
        "ความร้อน"
        in prompt
    )

    print(
        "✓ สร้าง RAG Prompt ผ่าน"
    )


# =========================================================
# Test 2: ทดสอบดึง Source ไม่ซ้ำ
# =========================================================

def test_extract_sources():

    results = [

        {
            "filename": "fire.txt"
        },

        {
            "filename": "fire.txt"
        },

        {
            "filename": "00_current_element.txt"
        },

    ]


    sources = extract_sources(
        results
    )


    assert sources == [

        "fire.txt",

        "00_current_element.txt",

    ]


    print(
        "✓ ดึง Sources ไม่ซ้ำผ่าน"
    )



# =========================================================
# Test 3: ทดสอบ Retrieval ภายใน RAG
# =========================================================

def test_retrieval_inside_rag():

    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=5,
    )


    results = rag.retrieve(

        question="ธาตุลมมีลักษณะอย่างไร",

        classification={

            "intent": "profile",

            "element": "wind",

        },

    )


    assert results


    assert any(

        "wind"
        in result["filename"].lower()

        for result in results

    )


    print(
        "✓ RAG เรียก Retrieval ผ่าน"
    )



# =========================================================
# Test 4: ทดสอบคำถามว่าง
# =========================================================

def test_empty_question():

    rag = RAGSystem(

        model=MODEL_NAME,

        top_k=5,

    )


    result = rag.answer(
        "   "
    )


    assert (
        result["sources"]
        == []
    )


    assert (
        result["retrieved_chunks"]
        == []
    )


    print(
        "✓ จัดการคำถามว่างผ่าน"
    )



# =========================================================
# Test 5: ทดสอบสร้างคำตอบ RAG
# =========================================================

def test_rag_answer():

    rag = RAGSystem(

        model=MODEL_NAME,

        top_k=5,

    )


    result = rag.answer(

        "ธาตุไฟมีลักษณะอย่างไร"

    )


    assert result["question"]


    assert result["answer"]


    assert (

        result.get("retrieved_chunks")

        or

        result.get("context")

    )


    print(
        "✓ RAG สร้างคำตอบผ่าน"
    )



# =========================================================
# Test 6: ทดสอบ Food Recommendation
# =========================================================

def test_food_recommendation():

    rag = RAGSystem(

        model=MODEL_NAME,

        top_k=5,

    )


    result = rag.answer(

        "ธาตุไฟควรกินผักอะไร"

    )


    assert result["answer"]


    assert result.get(
        "recommendation"
    )


    assert isinstance(

        result["recommendation"],

        dict

    )


    print(
        "✓ Food Recommendation ผ่าน"
    )



# =========================================================
# Test 7: ทดสอบ Menu Recommendation
# =========================================================

def test_menu_recommendation():

    rag = RAGSystem(

        model=MODEL_NAME,

        top_k=5,

    )


    result = rag.answer(

        "ธาตุลมมีเมนูอะไรบ้าง"

    )


    assert result["answer"]


    assert result.get(
        "recommendation"
    )


    assert isinstance(

        result["recommendation"],

        dict

    )


    print(
        "✓ Menu Recommendation ผ่าน"
    )



# =========================================================
# Test 8: ทดสอบ Warning
# =========================================================

def test_warning_recommendation():

    rag = RAGSystem(

        model=MODEL_NAME,

        top_k=5,

    )


    result = rag.answer(

        "ธาตุน้ำควรหลีกเลี่ยงอาหารอะไร"

    )


    assert result["answer"]


    assert result.get(
        "recommendation"
    ) is not None or result["answer"]


    print(
        "✓ Warning Recommendation ผ่าน"
    )



# =========================================================
# Run Test
# =========================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 60
    )

    print(
        "ทดสอบระบบ Retrieval-Augmented Generation"
    )

    print(
        "=" * 60
    )


    test_build_prompt()

    test_extract_sources()

    test_retrieval_inside_rag()

    test_empty_question()

    test_rag_answer()

    test_food_recommendation()

    test_menu_recommendation()

    test_warning_recommendation()


    print()

    print(
        "=" * 60
    )

    print(
        "✓ ทดสอบระบบ RAG สำเร็จทั้งหมด"
    )

    print(
        "=" * 60
    )