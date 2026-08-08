from retrieval import (
    TfidfRetriever,
    build_context,
    load_documents,
    split_into_chunks,
)


# =========================================================
# โหลดเอกสาร
# =========================================================

def test_load_documents():

    documents = load_documents()

    filenames = {
        document["filename"]
        for document in documents
    }

    required_files = {
        "00_current_element.txt",
        "earth.txt",
        "water.txt",
        "wind.txt",
        "fire.txt",
        "mixed_elements.txt",
    }

    missing = (
        required_files - filenames
    )

    assert not missing, (
        f"ขาดไฟล์ knowledge: {missing}"
    )

    print(
        "✓ โหลดฐานเอกสารความรู้ผ่าน"
    )


# =========================================================
# Chunk
# =========================================================

def test_split_chunks():

    documents = load_documents()

    chunks = split_into_chunks(
        documents
    )

    assert len(chunks) > len(
        documents
    )

    for chunk in chunks:

        assert chunk["chunk_id"]
        assert chunk["document_id"]
        assert chunk["filename"]
        assert chunk["text"]

    print(
        f"✓ แบ่งเอกสารเป็น "
        f"{len(chunks)} chunks ผ่าน"
    )


# =========================================================
# Current element
# =========================================================

def test_current_element_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "ธาตุเจ้าเรือนปัจจุบันคืออะไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "00_current_element"
        for result in results
    )

    print(
        "✓ ค้นความหมายธาตุเจ้าเรือนปัจจุบันผ่าน"
    )


# =========================================================
# Earth
# =========================================================

def test_earth_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "คนธาตุดินมีลักษณะนิสัยอย่างไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "earth"
        for result in results
    )

    print(
        "✓ ค้นข้อมูลธาตุดินผ่าน"
    )


# =========================================================
# Water
# =========================================================

def test_water_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "เสมหะมาก น้ำมูกมาก "
        "บวมง่าย เกี่ยวข้องกับธาตุอะไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "water"
        for result in results
    )

    print(
        "✓ ค้นข้อมูลธาตุน้ำผ่าน"
    )


# =========================================================
# Wind
# =========================================================

def test_wind_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "ท้องอืด มีลม เวียนหัว "
        "นอนไม่คงที่ เกี่ยวข้องกับธาตุอะไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "wind"
        for result in results
    )

    print(
        "✓ ค้นข้อมูลธาตุลมผ่าน"
    )


# =========================================================
# Fire
# =========================================================

def test_fire_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "ร้อนง่าย หิวเร็ว กระหายน้ำ "
        "หงุดหงิดง่าย เป็นลักษณะของธาตุอะไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "fire"
        for result in results
    )

    print(
        "✓ ค้นข้อมูลธาตุไฟผ่าน"
    )


# =========================================================
# Food
# =========================================================

def test_fire_food_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "ธาตุไฟควรกินอาหารอะไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "fire"
        for result in results
    )

    print(
        "✓ ค้นอาหารธาตุไฟผ่าน"
    )


def test_water_avoid_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "ธาตุน้ำควรหลีกเลี่ยง"
        "หรือระวังอาหารอะไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "water"
        for result in results
    )

    print(
        "✓ ค้นอาหารที่ควรระวัง"
        "ของธาตุน้ำผ่าน"
    )


# =========================================================
# Mixed
# =========================================================

def test_mixed_search():

    retriever = TfidfRetriever()

    results = retriever.search(
        "ถ้าสองธาตุเด่นใกล้เคียงกัน"
        "ควรแปลผลอย่างไร",
        top_k=5,
    )

    assert results

    assert any(
        result["document_id"]
        == "mixed_elements"
        for result in results
    )

    print(
        "✓ ค้นข้อมูลกรณีสองธาตุเด่นผ่าน"
    )


# =========================================================
# Build context
# =========================================================

def test_build_context():

    retriever = TfidfRetriever()

    results = retriever.search(
        "คนธาตุลมควรกินอะไร",
        top_k=3,
    )

    context = build_context(
        results
    )

    assert context
    assert "แหล่งข้อมูล:" in context

    print(
        "✓ สร้าง Context สำหรับ RAG ผ่าน"
    )


# =========================================================
# Empty query
# =========================================================

def test_empty_query():

    retriever = TfidfRetriever()

    results = retriever.search(
        "",
        top_k=5,
    )

    assert results == []

    print(
        "✓ จัดการคำถามว่างผ่าน"
    )


# =========================================================
# Run tests
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "ทดสอบระบบ Information Retrieval"
    )
    print("=" * 60)

    test_load_documents()
    test_split_chunks()

    test_current_element_search()

    test_earth_search()
    test_water_search()
    test_wind_search()
    test_fire_search()

    test_fire_food_search()
    test_water_avoid_search()

    test_mixed_search()

    test_build_context()
    test_empty_query()

    print()

    print("=" * 60)

    print(
        "✓ ทดสอบระบบ Information "
        "Retrieval สำเร็จทั้งหมด"
    )

    print("=" * 60)