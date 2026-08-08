from __future__ import annotations

from rag import (
    RAGSystem,
    build_rag_prompt,
    extract_sources,
)


def test_build_prompt():
    prompt = build_rag_prompt(
        question="ธาตุไฟมีลักษณะอย่างไร",
        context="ธาตุไฟสัมพันธ์กับความร้อนและการย่อยอาหาร",
    )

    assert "ธาตุไฟมีลักษณะอย่างไร" in prompt
    assert "ความร้อนและการย่อยอาหาร" in prompt

    print("✓ สร้าง RAG Prompt ผ่าน")


def test_extract_sources():
    results = [
        {"filename": "fire.txt"},
        {"filename": "fire.txt"},
        {"filename": "00_current_element.txt"},
    ]

    sources = extract_sources(results)

    assert sources == [
        "fire.txt",
        "00_current_element.txt",
    ]

    print("✓ ดึง Sources ไม่ซ้ำผ่าน")


def test_retrieval_inside_rag():
    rag = RAGSystem(
        model="scb10x/typhoon2.5-qwen3-4b:latest",
        top_k=5,
    )

    results = rag.retrieve(
        "ธาตุลมมีลักษณะอย่างไร"
    )

    assert results

    assert any(
        result["document_id"] == "wind"
        for result in results
    )

    print("✓ RAG เรียก Retrieval ผ่าน")


def test_empty_question():
    rag = RAGSystem(
        model="scb10x/typhoon2.5-qwen3-4b:latest",
        top_k=5,
    )

    result = rag.answer("   ")

    assert result["sources"] == []
    assert result["retrieved_chunks"] == []

    print("✓ จัดการคำถามว่างผ่าน")


def test_rag_answer():
    rag = RAGSystem(
        model="scb10x/typhoon2.5-qwen3-4b:latest",
        top_k=5,
    )

    result = rag.answer(
        "ธาตุไฟควรกินอาหารอะไร"
    )

    assert result["question"]
    assert result["answer"]
    assert result["retrieved_chunks"]

    print("✓ RAG สร้างคำตอบผ่าน")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("ทดสอบระบบ Retrieval-Augmented Generation")
    print("=" * 60)

    test_build_prompt()
    test_extract_sources()
    test_retrieval_inside_rag()
    test_empty_question()
    test_rag_answer()

    print()
    print("=" * 60)
    print("✓ ทดสอบระบบ RAG สำเร็จทั้งหมด")
    print("=" * 60)