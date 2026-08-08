from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Path
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


# =========================================================
# ค่าพื้นฐาน
# =========================================================

ALLOWED_EXTENSIONS = {".txt"}

DEFAULT_TOP_K = 5

MIN_SIMILARITY = 0.01


# =========================================================
# โหลดเอกสาร
# =========================================================

def load_documents() -> list[dict[str, str]]:
    """
    โหลดไฟล์ .txt ทั้งหมดจาก knowledge/

    คืนค่า:
    [
        {
            "document_id": "earth",
            "filename": "earth.txt",
            "text": "..."
        },
        ...
    ]
    """

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์ knowledge: {KNOWLEDGE_DIR}"
        )

    documents: list[dict[str, str]] = []

    for file_path in sorted(KNOWLEDGE_DIR.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append(
            {
                "document_id": file_path.stem,
                "filename": file_path.name,
                "text": text,
            }
        )

    if not documents:
        raise ValueError(
            "ไม่พบเอกสารความรู้ .txt ในโฟลเดอร์ knowledge"
        )

    return documents


# =========================================================
# Chunking
# =========================================================

def split_into_chunks(
    documents: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    แบ่งเอกสารเป็น chunk ตามย่อหน้า

    ในไฟล์ knowledge เราใช้บรรทัดว่างคั่นแต่ละหัวข้อ
    จึงสามารถแบ่งด้วย \\n\\n ได้โดยตรง
    """

    chunks: list[dict[str, str]] = []

    for document in documents:

        raw_sections = document["text"].split("\n\n")

        sections = [
            section.strip()
            for section in raw_sections
            if section.strip()
        ]

        for index, section in enumerate(
            sections,
            start=1,
        ):

            chunks.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}"
                        f"_chunk_{index:03d}"
                    ),
                    "document_id": document[
                        "document_id"
                    ],
                    "filename": document[
                        "filename"
                    ],
                    "text": section,
                }
            )

    if not chunks:
        raise ValueError(
            "ไม่สามารถสร้าง chunk จากฐานความรู้ได้"
        )

    return chunks


# =========================================================
# Retriever
# =========================================================

class TfidfRetriever:
    """
    Information Retrieval แบบ TF-IDF
    + Cosine Similarity

    ใช้ character n-gram เพื่อให้เหมาะกับข้อความภาษาไทย
    ที่ไม่มีการเว้นวรรคระหว่างคำทุกคำ
    """

    def __init__(self) -> None:

        self.documents = load_documents()

        self.chunks = split_into_chunks(
            self.documents
        )

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
        )

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        self.document_matrix = (
            self.vectorizer.fit_transform(texts)
        )


    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> list[dict[str, Any]]:
        """
        ค้น chunk ที่มีความใกล้เคียงกับคำถาม

        Parameters
        ----------
        query:
            คำถามของผู้ใช้

        top_k:
            จำนวนผลลัพธ์สูงสุด

        min_similarity:
            similarity ขั้นต่ำที่จะยอมรับ
        """

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k ต้องมากกว่า 0"
            )

        if not 0 <= min_similarity <= 1:
            raise ValueError(
                "min_similarity ต้องอยู่ระหว่าง 0 ถึง 1"
            )

        query_vector = (
            self.vectorizer.transform([query])
        )

        similarities = cosine_similarity(
            query_vector,
            self.document_matrix,
        )[0]

        ranked_indices = (
            similarities.argsort()[::-1]
        )

        results: list[dict[str, Any]] = []

        for index in ranked_indices:

            similarity = float(
                similarities[index]
            )

            if similarity < min_similarity:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "rank": len(results) + 1,
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "document_id": chunk[
                        "document_id"
                    ],
                    "filename": chunk[
                        "filename"
                    ],
                    "text": chunk[
                        "text"
                    ],
                    "score": round(
                        similarity,
                        4,
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results


# =========================================================
# ฟังก์ชันช่วยสำหรับ RAG ในอนาคต
# =========================================================

def build_context(
    results: list[dict[str, Any]],
) -> str:
    """
    รวมผลจาก IR เป็น context

    ฟังก์ชันนี้เตรียมไว้สำหรับขั้น RAG ต่อไป
    """

    if not results:
        return ""

    context_parts: list[str] = []

    for result in results:

        context_parts.append(
            (
                f"[แหล่งข้อมูล: "
                f"{result['filename']} | "
                f"{result['chunk_id']}]\n"
                f"{result['text']}"
            )
        )

    return "\n\n".join(context_parts)


# =========================================================
# แสดงผลใน Terminal
# =========================================================

def print_search_results(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> None:

    retriever = TfidfRetriever()

    results = retriever.search(
        query=query,
        top_k=top_k,
    )

    print("=" * 70)

    print(
        "คำถาม:",
        query,
    )

    print("=" * 70)

    if not results:

        print(
            "ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้"
        )

        return

    for result in results:

        print()

        print(
            f"อันดับ {result['rank']}"
        )

        print(
            f"เอกสาร: {result['filename']}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print("-" * 70)

        print(
            result["text"]
        )

        print("-" * 70)


# =========================================================
# ทดสอบแบบ Manual
# =========================================================

if __name__ == "__main__":

    test_queries = [
        "ธาตุเจ้าเรือนปัจจุบันคืออะไร",
        "คนธาตุลมมีอาการอะไรบ้าง",
        "ธาตุไฟควรกินอาหารอะไร",
        "ธาตุน้ำควรหลีกเลี่ยงอาหารอะไร",
        "คนธาตุดินมีลักษณะนิสัยอย่างไร",
    ]

    for query in test_queries:

        print_search_results(
            query=query,
            top_k=5,
        )

        print("\n")