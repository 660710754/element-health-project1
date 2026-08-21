from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

DEFAULT_TOP_K = 5

# Retrieval weights
EXPLICIT_ELEMENT_BOOST = 0.30
PRIMARY_ELEMENT_BOOST = 0.10
SECONDARY_ELEMENT_BOOST = 0.05
KNOWLEDGE_TYPE_BOOST = 0.15

# Chunk configuration
MAX_CHUNK_CHARS = 500
MIN_USEFUL_CHUNK_CHARS = 25


# =========================================================
# Element Mapping
# =========================================================

ELEMENTS = [
    "earth",
    "water",
    "wind",
    "fire",
]


ELEMENT_NAMES_TH = {
    "earth": "ธาตุดิน",
    "water": "ธาตุน้ำ",
    "wind": "ธาตุลม",
    "fire": "ธาตุไฟ",
}


# =========================================================
# Knowledge Categories
# =========================================================

KNOWLEDGE_TYPES = [
    "profile",
    "food",
    "menu",
    "warning",
]


KNOWLEDGE_TYPE_NAMES_TH = {
    "profile": "ลักษณะธาตุ",
    "food": "อาหาร",
    "menu": "เมนู",
    "warning": "ข้อควรระวัง",
}


# =========================================================
# Intent Keywords
# =========================================================

INTENT_KEYWORDS = {

    "warning": [
        "หลีกเลี่ยง",
        "ควรระวัง",
        "ระวัง",
        "ไม่ควรกิน",
        "ไม่ควรรับประทาน",
        "ข้อควรระวัง",
    ],

    "menu": [
        "เมนู",
        "เมนูอาหาร",
        "ทำอาหาร",
        "เมนูแนะนำ",
        "เมนูอะไร",
    ],

    "food": [
        "ควรกิน",
        "กินอะไร",
        "ควรรับประทาน",
        "อาหารอะไร",
        "อาหารที่เหมาะ",
        "อาหารแนะนำ",
        "ควรกินอะไร",
    ],

    "profile": [
        "ลักษณะ",
        "ลักษณะธาตุ",
        "เป็นอย่างไร",
        "อาการ",
        "บุคลิก",
        "นิสัย",
    ],
}


# =========================================================
# Load Text File
# =========================================================

def load_text_file(
    file_path: Path,
) -> str:
    """
    อ่านไฟล์ knowledge .txt
    """

    if not file_path.exists():
        return ""

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return file.read().strip()


# =========================================================
# Validation
# =========================================================

def validate_element(
    element: str,
) -> None:
    """
    ตรวจสอบชื่อธาตุ
    """

    if element not in ELEMENTS:
        raise ValueError(
            f"Unknown element: {element}"
        )


# =========================================================
# Detect Explicit Elements
# =========================================================

def detect_explicit_elements(
    query: str,
) -> list[str]:
    """
    ตรวจว่าผู้ใช้ระบุชื่อธาตุใดในคำถามโดยตรง

    ตัวอย่าง:
    "ธาตุน้ำควรกินอะไร"
    -> ["water"]

    "ธาตุน้ำและธาตุดินควรกินอะไร"
    -> ["earth", "water"]
    """

    detected: list[str] = []


    for element in ELEMENTS:

        element_name_th = (
            ELEMENT_NAMES_TH[element]
        )

        if element_name_th in query:

            detected.append(
                element
            )


    return detected


# =========================================================
# Detect Query Intents
# =========================================================

def detect_query_intents(
    query: str,
) -> list[str]:
    """
    ตรวจ intent ของคำถาม

    รองรับมากกว่า 1 intent

    ตัวอย่าง:
    "ควรกินอะไรและควรหลีกเลี่ยงอะไร"

    -> ["warning", "food"]
    """

    detected: list[str] = []


    for knowledge_type, keywords in (
        INTENT_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in query:

                detected.append(
                    knowledge_type
                )

                break


    return detected


# =========================================================
# Load Element Knowledge
# =========================================================

def load_element_knowledge(
    element: str,
) -> dict[str, str]:
    """
    โหลด knowledge ของธาตุหนึ่ง
    """

    validate_element(
        element
    )


    element_dir = (
        KNOWLEDGE_DIR
        /
        element
    )


    knowledge: dict[str, str] = {}


    for knowledge_type in KNOWLEDGE_TYPES:

        file_path = (
            element_dir
            /
            f"{element}_{knowledge_type}.txt"
        )


        knowledge[
            knowledge_type
        ] = load_text_file(
            file_path
        )


    return knowledge


# =========================================================
# Question-like Chunk Detection
# =========================================================

def is_question_like_chunk(
    text: str,
) -> bool:
    """
    ตรวจ chunk ที่ดูเหมือนเป็นตัวอย่างคำถาม
    มากกว่าเนื้อหาความรู้

    เช่น:
    "ธาตุดินควรหลีกเลี่ยงอาหารอะไร"

    ไม่ควรนำไปเป็น RAG context หลัก
    """

    cleaned = (
        text
        .strip()
        .strip("\"'")
    )


    question_patterns = [
        "อะไร",
        "อย่างไร",
        "แบบไหน",
        "หรือไม่",
        "ไหม",
        "มั้ย",
    ]


    if len(cleaned) <= 100:

        for pattern in question_patterns:

            if pattern in cleaned:
                return True


    return False


# =========================================================
# Heading-like Detection
# =========================================================

def is_heading_like(
    text: str,
) -> bool:
    """
    ตรวจข้อความที่มีลักษณะเป็นหัวข้อสั้น ๆ

    ตัวอย่าง:
    ธาตุน้ำ:
    ธาตุน้ำ (อาโปธาตุ)
    อาหารที่เหมาะสม
    ข้อควรระวัง

    หัวข้อเหล่านี้ไม่ควรถูกสร้างเป็น chunk เดี่ยว
    """

    cleaned = (
        text
        .strip()
        .strip("\"'")
    )


    if not cleaned:
        return False


    # ยาวเกินไป ไม่น่าจะเป็น heading
    if len(cleaned) > 80:
        return False


    # ถ้ามีหลายบรรทัดมาก
    # ไม่น่าจะเป็น heading
    lines = [
        line
        for line in cleaned.splitlines()
        if line.strip()
    ]


    if len(lines) > 2:
        return False


    # หัวข้อที่พบบ่อยใน knowledge
    heading_keywords = [
        "ธาตุดิน",
        "ธาตุน้ำ",
        "ธาตุลม",
        "ธาตุไฟ",
        "ปถวีธาตุ",
        "อาโปธาตุ",
        "วาโยธาตุ",
        "เตโชธาตุ",
        "อาหาร",
        "เมนู",
        "ผัก",
        "ผลไม้",
        "เครื่องดื่ม",
        "ข้อควรระวัง",
        "ลักษณะ",
        "รส",
    ]


    # ลงท้ายด้วย :
    if cleaned.endswith(":"):
        return True


    # ข้อความสั้นมากและมี keyword แบบหัวข้อ
    if len(cleaned) <= 40:

        for keyword in heading_keywords:

            if keyword in cleaned:
                return True


    return False


# =========================================================
# Useful Chunk Detection
# =========================================================

def is_useful_chunk(
    text: str,
) -> bool:
    """
    ตรวจว่า chunk มีข้อมูลมากพอสำหรับ retrieval/RAG หรือไม่
    """

    cleaned = (
        text
        .strip()
        .strip("\"'")
    )


    if not cleaned:
        return False


    # ไม่เอาตัวอย่างคำถาม
    if is_question_like_chunk(
        cleaned
    ):
        return False


    # ไม่เอาหัวข้อเดี่ยว
    if is_heading_like(
        cleaned
    ):
        return False


    # สั้นเกินไป
    if len(cleaned) < MIN_USEFUL_CHUNK_CHARS:
        return False


    return True


# =========================================================
# Split Large Text
# =========================================================

def split_large_part(
    text: str,
    max_chars: int,
) -> list[str]:
    """
    แบ่งข้อความที่ยาวเกิน max_chars
    โดยพยายามแบ่งตามบรรทัด
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


    chunks: list[str] = []

    current_chunk = ""


    for line in lines:

        candidate = (
            f"{current_chunk}\n{line}"
        ).strip()


        if (
            current_chunk
            and len(candidate) > max_chars
        ):

            chunks.append(
                current_chunk
            )

            current_chunk = line

        else:

            current_chunk = candidate


    if current_chunk:

        chunks.append(
            current_chunk
        )


    return chunks


# =========================================================
# Chunk Knowledge
# =========================================================

def split_text_into_chunks(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """
    แบ่ง knowledge เป็น chunks

    หลักการ:
    1. ใช้บรรทัดว่างแบ่ง section
    2. ถ้า section เป็น heading สั้น ๆ
       ให้รวมกับ section ถัดไป
    3. ไม่สร้าง heading เป็น chunk เดี่ยว
    4. ไม่เอาตัวอย่างคำถาม
    5. กรอง chunk ที่สั้นเกินไป
    """

    if not text:
        return []


    raw_parts = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]


    merged_parts: list[str] = []

    pending_heading = ""


    # -----------------------------------------------------
    # Merge Heading with Following Content
    # -----------------------------------------------------

    for part in raw_parts:

        # ตัวอย่างคำถาม
        # ไม่เอาเข้าฐาน retrieval
        if is_question_like_chunk(
            part
        ):
            continue


        if is_heading_like(
            part
        ):

            if pending_heading:

                pending_heading = (
                    f"{pending_heading}\n{part}"
                )

            else:

                pending_heading = part

            continue


        if pending_heading:

            merged = (
                f"{pending_heading}\n{part}"
            ).strip()

            merged_parts.append(
                merged
            )

            pending_heading = ""

        else:

            merged_parts.append(
                part
            )


    # ถ้าเหลือ heading ตัวเดียวท้ายไฟล์
    # ไม่สร้างเป็น chunk เพราะไม่มีเนื้อหา
    pending_heading = ""


    # -----------------------------------------------------
    # Split Oversized Sections
    # -----------------------------------------------------

    chunks: list[str] = []


    for part in merged_parts:

        if len(part) <= max_chars:

            if is_useful_chunk(
                part
            ):

                chunks.append(
                    part
                )

            continue


        large_chunks = (
            split_large_part(
                text=part,
                max_chars=max_chars,
            )
        )


        for chunk in large_chunks:

            if is_useful_chunk(
                chunk
            ):

                chunks.append(
                    chunk
                )


    return chunks


# =========================================================
# Build Knowledge Documents
# =========================================================

def build_knowledge_documents(
    elements: list[str],
) -> list[dict[str, Any]]:
    """
    เปลี่ยน knowledge files
    เป็น retrieval documents

    1 document = 1 useful chunk
    """

    documents: list[
        dict[str, Any]
    ] = []


    for element in elements:

        validate_element(
            element
        )


        element_knowledge = (
            load_element_knowledge(
                element
            )
        )


        for knowledge_type, text in (
            element_knowledge.items()
        ):

            if not text:
                continue


            chunks = split_text_into_chunks(
                text
            )


            for chunk_index, chunk in enumerate(
                chunks,
                start=1
            ):

                documents.append(
                    {
                        "element": element,

                        "element_th": (
                            ELEMENT_NAMES_TH[
                                element
                            ]
                        ),

                        "knowledge_type": (
                            knowledge_type
                        ),

                        "knowledge_type_th": (
                            KNOWLEDGE_TYPE_NAMES_TH[
                                knowledge_type
                            ]
                        ),

                        "source_file": (
                            f"{element}_"
                            f"{knowledge_type}.txt"
                        ),

                        "chunk_id": (
                            f"{element}_"
                            f"{knowledge_type}_"
                            f"{chunk_index}"
                        ),

                        "text": chunk,
                    }
                )


    return documents


# =========================================================
# Calculate Element Boost
# =========================================================

def calculate_element_boost(
    element: str,
    active_elements: list[str],
    explicit_elements: list[str],
) -> float:
    """
    คำนวณน้ำหนักของธาตุ

    ลำดับความสำคัญ:
    1. ธาตุที่ผู้ใช้ระบุในคำถามโดยตรง
    2. Primary element
    3. Secondary element
    """

    boost = 0.0


    # -----------------------------------------------------
    # Explicit Element
    # -----------------------------------------------------

    if element in explicit_elements:

        boost += (
            EXPLICIT_ELEMENT_BOOST
        )


    # -----------------------------------------------------
    # Primary Element
    # -----------------------------------------------------

    if active_elements:

        if element == active_elements[0]:

            boost += (
                PRIMARY_ELEMENT_BOOST
            )


    # -----------------------------------------------------
    # Secondary Element
    # -----------------------------------------------------

    if len(active_elements) >= 2:

        if element == active_elements[1]:

            boost += (
                SECONDARY_ELEMENT_BOOST
            )


    return boost


# =========================================================
# Calculate Knowledge Type Boost
# =========================================================

def calculate_type_boost(
    knowledge_type: str,
    query_intents: list[str],
) -> float:
    """
    เพิ่มคะแนนให้ knowledge type
    ที่ตรงกับ intent ของคำถาม
    """

    if knowledge_type in query_intents:

        return (
            KNOWLEDGE_TYPE_BOOST
        )


    return 0.0


# =========================================================
# Retrieve Knowledge
# =========================================================

def retrieve_knowledge(
    query: str,
    active_elements: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Hybrid Knowledge Retrieval

    Final Score =
        TF-IDF Similarity
        + Element Boost
        + Knowledge Type Boost

    active_elements:
    index 0 = primary
    index 1 = secondary (ถ้ามี)
    """

    if top_k <= 0:

        raise ValueError(
            "top_k ต้องมากกว่า 0"
        )


    query = query.strip()


    if not query:
        return []


    if not active_elements:
        return []


    for element in active_elements:

        validate_element(
            element
        )


    # -----------------------------------------------------
    # Query Analysis
    # -----------------------------------------------------

    explicit_elements = (
        detect_explicit_elements(
            query
        )
    )


    query_intents = (
        detect_query_intents(
            query
        )
    )


    # -----------------------------------------------------
    # Build Retrieval Documents
    # -----------------------------------------------------

    documents = (
        build_knowledge_documents(
            active_elements
        )
    )


    if not documents:
        return []


    corpus = [
        document["text"]
        for document
        in documents
    ]


    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
    )


    try:

        document_matrix = (
            vectorizer.fit_transform(
                corpus
            )
        )


        query_vector = (
            vectorizer.transform(
                [query]
            )
        )


    except ValueError:

        return []


    similarities = cosine_similarity(
        query_vector,
        document_matrix,
    )[0]


    # -----------------------------------------------------
    # Hybrid Ranking
    # -----------------------------------------------------

    ranked_results: list[
        dict[str, Any]
    ] = []


    for index, similarity in enumerate(
        similarities
    ):

        document = dict(
            documents[index]
        )


        tfidf_score = float(
            similarity
        )


        element_boost = (
            calculate_element_boost(
                element=document[
                    "element"
                ],
                active_elements=(
                    active_elements
                ),
                explicit_elements=(
                    explicit_elements
                ),
            )
        )


        type_boost = (
            calculate_type_boost(
                knowledge_type=document[
                    "knowledge_type"
                ],
                query_intents=(
                    query_intents
                ),
            )
        )


        final_score = (
            tfidf_score
            + element_boost
            + type_boost
        )


        document[
            "tfidf_score"
        ] = round(
            tfidf_score,
            4
        )


        document[
            "element_boost"
        ] = round(
            element_boost,
            4
        )


        document[
            "type_boost"
        ] = round(
            type_boost,
            4
        )


        document[
            "retrieval_score"
        ] = round(
            final_score,
            4
        )


        ranked_results.append(
            document
        )


    # -----------------------------------------------------
    # Sort Results
    # -----------------------------------------------------

    ranked_results.sort(
        key=lambda item: (
            -item[
                "retrieval_score"
            ],
            -item[
                "tfidf_score"
            ],
            item[
                "source_file"
            ],
            item[
                "chunk_id"
            ],
        )
    )


    return ranked_results[
        :top_k
    ]


# =========================================================
# Build RAG Context
# =========================================================

def build_knowledge_context(
    query: str,
    active_elements: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """
    สร้าง RAG Context
    จาก Top-K Knowledge Retrieval Results
    """

    results = retrieve_knowledge(
        query=query,
        active_elements=active_elements,
        top_k=top_k,
    )


    if not results:
        return ""


    context_parts: list[str] = []


    for index, item in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            (
                f"[Context {index}]\n"
                f"ธาตุ: "
                f"{item['element_th']}\n"
                f"ประเภท: "
                f"{item['knowledge_type_th']}\n"
                f"แหล่งข้อมูล: "
                f"{item['source_file']}\n"
                f"Retrieval Score: "
                f"{item['retrieval_score']}\n\n"
                f"{item['text']}"
            )
        )


    return "\n\n".join(
        context_parts
    )


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "Hybrid Knowledge Retrieval Test"
    )

    print(
        "=" * 70
    )


    test_query = (
        "ธาตุน้ำควรกินอะไร "
        "และควรหลีกเลี่ยงอะไร"
    )


    # ตัวแรก = primary
    # ตัวที่สอง = secondary
    test_elements = [
        "water",
        "earth",
    ]


    print(
        f"QUERY: {test_query}"
    )


    print(
        "ACTIVE ELEMENTS: "
        + ", ".join(
            test_elements
        )
    )


    explicit_elements = (
        detect_explicit_elements(
            test_query
        )
    )


    query_intents = (
        detect_query_intents(
            test_query
        )
    )


    print(
        "EXPLICIT ELEMENTS: "
        + (
            ", ".join(
                explicit_elements
            )
            if explicit_elements
            else "-"
        )
    )


    print(
        "QUERY INTENTS: "
        + (
            ", ".join(
                query_intents
            )
            if query_intents
            else "-"
        )
    )


    print()


    results = retrieve_knowledge(
        query=test_query,
        active_elements=test_elements,
        top_k=5,
    )


    for index, item in enumerate(
        results,
        start=1
    ):

        print(
            "-" * 70
        )


        print(
            f"{index}. "
            f"{item['source_file']}"
        )


        print(
            f"   element = "
            f"{item['element']}"
        )


        print(
            f"   type = "
            f"{item['knowledge_type']}"
        )


        print(
            f"   tfidf = "
            f"{item['tfidf_score']}"
        )


        print(
            f"   element_boost = "
            f"{item['element_boost']}"
        )


        print(
            f"   type_boost = "
            f"{item['type_boost']}"
        )


        print(
            f"   final_score = "
            f"{item['retrieval_score']}"
        )


        print(
            f"   text = "
            f"{item['text'][:300]}"
        )


    print()

    print(
        "=" * 70
    )

    print(
        "RAG Context"
    )

    print(
        "=" * 70
    )


    context = build_knowledge_context(
        query=test_query,
        active_elements=test_elements,
        top_k=5,
    )


    print(
        context
    )