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

WARNING_KEYWORDS = [
    "หลีกเลี่ยง",
    "ควรระวัง",
    "ระวัง",
    "ไม่ควรกิน",
    "ไม่ควรรับประทาน",
    "ข้อควรระวัง",
]


MENU_KEYWORDS = [
    "เมนู",
    "เมนูอาหาร",
    "ทำอาหาร",
    "เมนูแนะนำ",
    "เมนูอะไร",
]


FOOD_KEYWORDS = [
    "ควรกิน",
    "กินอะไร",
    "ควรรับประทาน",
    "อาหารที่เหมาะ",
    "อาหารแนะนำ",
]


PROFILE_KEYWORDS = [
    "ลักษณะ",
    "ลักษณะธาตุ",
    "เป็นอย่างไร",
    "อาการ",
    "บุคลิก",
    "นิสัย",
]


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
# Keyword Detection Helper
# =========================================================

def contains_any_keyword(
    query: str,
    keywords: list[str],
) -> bool:
    """
    ตรวจว่า query มี keyword
    อย่างน้อยหนึ่งคำหรือไม่
    """

    return any(
        keyword in query
        for keyword in keywords
    )


# =========================================================
# Detect Query Intents
# =========================================================

def detect_query_intents(
    query: str,
) -> list[str]:
    """
    ตรวจ intent ของคำถาม

    รองรับหลาย intent เช่น:
    "ธาตุน้ำควรกินอะไร และควรหลีกเลี่ยงอะไร"
    -> ["warning", "food"]

    Intent Priority:
    - คำว่า "หลีกเลี่ยงอาหารอะไร"
      ถือเป็น warning ไม่ใช่ food
    - food จะถูกตรวจจากคำเชิงแนะนำ
      เช่น "ควรกิน", "กินอะไร"
    """

    query = query.strip()

    if not query:
        return []

    detected: list[str] = []

    has_warning = contains_any_keyword(
        query,
        WARNING_KEYWORDS,
    )

    has_menu = contains_any_keyword(
        query,
        MENU_KEYWORDS,
    )

    has_food = contains_any_keyword(
        query,
        FOOD_KEYWORDS,
    )

    has_profile = contains_any_keyword(
        query,
        PROFILE_KEYWORDS,
    )


    # -----------------------------------------------------
    # Warning
    # -----------------------------------------------------

    if has_warning:

        detected.append(
            "warning"
        )


    # -----------------------------------------------------
    # Menu
    # -----------------------------------------------------

    if has_menu:

        detected.append(
            "menu"
        )


    # -----------------------------------------------------
    # Food
    #
    # สำคัญ:
    # ไม่ใช้คำกว้าง ๆ เช่น "อาหารอะไร"
    # เพราะจะทำให้:
    #
    # "ควรหลีกเลี่ยงอาหารอะไร"
    #
    # ถูกตีเป็น warning + food
    # -----------------------------------------------------

    if has_food:

        detected.append(
            "food"
        )


    # -----------------------------------------------------
    # Profile
    # -----------------------------------------------------

    if has_profile:

        detected.append(
            "profile"
        )


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
    เพื่อไม่ให้สร้างเป็น chunk เดี่ยว
    """

    cleaned = (
        text
        .strip()
        .strip("\"'")
    )

    if not cleaned:
        return False

    if len(cleaned) > 80:
        return False

    lines = [
        line
        for line in cleaned.splitlines()
        if line.strip()
    ]

    if len(lines) > 2:
        return False

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

    if cleaned.endswith(":"):
        return True

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
    ตรวจว่า chunk มีข้อมูลมากพอ
    สำหรับ Retrieval / RAG หรือไม่
    """

    cleaned = (
        text
        .strip()
        .strip("\"'")
    )

    if not cleaned:
        return False

    if is_question_like_chunk(
        cleaned
    ):
        return False

    if is_heading_like(
        cleaned
    ):
        return False

    if (
        len(cleaned)
        < MIN_USEFUL_CHUNK_CHARS
    ):
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
    แบ่ง Knowledge เป็น chunks

    หลักการ:
    1. ใช้บรรทัดว่างแบ่ง section
    2. ถ้าเป็น heading ให้รวมกับ section ถัดไป
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

        large_chunks = split_large_part(
            text=part,
            max_chars=max_chars,
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
    เปลี่ยน Knowledge files
    เป็น Retrieval Documents

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
                start=1,
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

    Priority:
    1. ธาตุที่ระบุในคำถามโดยตรง
    2. Primary element
    3. Secondary element
    """

    boost = 0.0


    # Explicit Element
    if element in explicit_elements:

        boost += (
            EXPLICIT_ELEMENT_BOOST
        )


    # Primary
    if active_elements:

        if (
            element
            == active_elements[0]
        ):

            boost += (
                PRIMARY_ELEMENT_BOOST
            )


    # Secondary
    if len(active_elements) >= 2:

        if (
            element
            == active_elements[1]
        ):

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
    เพิ่มคะแนนให้ Knowledge Type
    ที่ตรงกับ Intent
    """

    if (
        knowledge_type
        in query_intents
    ):

        return (
            KNOWLEDGE_TYPE_BOOST
        )

    return 0.0


# =========================================================
# Find Best Candidate for Element
# =========================================================

def find_best_element_candidate(
    ranked_results: list[dict[str, Any]],
    element: str,
    query_intents: list[str],
    selected_chunk_ids: set[str],
) -> dict[str, Any] | None:
    """
    หา candidate ที่ดีที่สุด
    สำหรับธาตุที่ยังไม่ปรากฏใน Top-K

    Priority:
    1. element ตรง
    2. knowledge_type ตรง intent
    3. retrieval_score สูง
    """

    candidates = [
        item
        for item in ranked_results
        if (
            item["element"] == element
            and item["chunk_id"]
            not in selected_chunk_ids
        )
    ]

    if not candidates:
        return None


    # -----------------------------------------------------
    # Prefer Type Matching Intent
    # -----------------------------------------------------

    if query_intents:

        intent_candidates = [
            item
            for item in candidates
            if (
                item["knowledge_type"]
                in query_intents
            )
        ]

        if intent_candidates:

            return intent_candidates[0]


    return candidates[0]


# =========================================================
# Mixed Element Coverage Safeguard
# =========================================================

def ensure_mixed_element_coverage(
    ranked_results: list[dict[str, Any]],
    active_elements: list[str],
    query_intents: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    ป้องกันกรณี Mixed Retrieval
    ที่ Top-K ถูกครองโดยธาตุเดียวทั้งหมด

    ตัวอย่าง:

    Active:
        water + earth

    Raw Top 5:
        water
        water
        water
        water
        water

    หลัง safeguard:
        water
        water
        water
        water
        earth

    หลักการ:
    - ใช้เฉพาะเมื่อมีมากกว่า 1 active element
    - ไม่เปลี่ยน Top-1 โดยไม่จำเป็น
    - เลือก candidate ที่ตรง intent ก่อน
    - แทนที่รายการท้าย ๆ จากธาตุที่มีมากกว่า 1 รายการ
    """

    if not ranked_results:
        return []

    selected = [
        dict(item)
        for item in ranked_results[
            :top_k
        ]
    ]

    if (
        len(active_elements) <= 1
        or top_k < len(
            set(active_elements)
        )
    ):
        return selected


    required_elements = list(
        dict.fromkeys(
            active_elements
        )
    )


    for required_element in (
        required_elements
    ):

        found_elements = [
            item["element"]
            for item in selected
        ]


        if (
            required_element
            in found_elements
        ):
            continue


        selected_chunk_ids = {
            item["chunk_id"]
            for item in selected
        }


        candidate = (
            find_best_element_candidate(
                ranked_results=(
                    ranked_results
                ),
                element=(
                    required_element
                ),
                query_intents=(
                    query_intents
                ),
                selected_chunk_ids=(
                    selected_chunk_ids
                ),
            )
        )


        if candidate is None:
            continue


        # -------------------------------------------------
        # Count Current Elements
        # -------------------------------------------------

        element_counts: dict[
            str,
            int
        ] = {}

        for item in selected:

            element = item[
                "element"
            ]

            element_counts[
                element
            ] = (
                element_counts.get(
                    element,
                    0,
                )
                + 1
            )


        # -------------------------------------------------
        # Find Replacement
        #
        # เริ่มจากท้าย Top-K
        # และเลือกธาตุที่มีมากกว่า 1 chunk
        # -------------------------------------------------

        replacement_index = None


        for index in range(
            len(selected) - 1,
            -1,
            -1,
        ):

            current_element = (
                selected[index][
                    "element"
                ]
            )


            if (
                element_counts.get(
                    current_element,
                    0,
                )
                > 1
            ):

                replacement_index = (
                    index
                )

                break


        if replacement_index is None:
            continue


        selected[
            replacement_index
        ] = dict(
            candidate
        )


    return selected


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

    หลัง Ranking:
        Mixed Element Coverage Safeguard

    active_elements:
        index 0 = primary
        index 1 = secondary
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


    # Remove duplicated element names
    active_elements = list(
        dict.fromkeys(
            active_elements
        )
    )


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
            4,
        )


        document[
            "element_boost"
        ] = round(
            element_boost,
            4,
        )


        document[
            "type_boost"
        ] = round(
            type_boost,
            4,
        )


        document[
            "retrieval_score"
        ] = round(
            final_score,
            4,
        )


        ranked_results.append(
            document
        )


    # -----------------------------------------------------
    # Sort Raw Ranking
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


    # -----------------------------------------------------
    # Mixed Element Coverage Safeguard
    # -----------------------------------------------------

    final_results = (
        ensure_mixed_element_coverage(
            ranked_results=(
                ranked_results
            ),
            active_elements=(
                active_elements
            ),
            query_intents=(
                query_intents
            ),
            top_k=top_k,
        )
    )


    return final_results


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
        active_elements=(
            active_elements
        ),
        top_k=top_k,
    )


    if not results:
        return ""


    context_parts: list[str] = []


    for index, item in enumerate(
        results,
        start=1,
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


    test_cases = [

        {
            "name": "Warning Intent Priority",

            "query": (
                "ธาตุน้ำควรหลีกเลี่ยงอาหารอะไร"
            ),

            "elements": [
                "water"
            ],
        },

        {
            "name": "Combined Food + Warning",

            "query": (
                "ธาตุน้ำควรกินอะไร "
                "และควรหลีกเลี่ยงอะไร"
            ),

            "elements": [
                "water"
            ],
        },

        {
            "name": "Mixed Element Food",

            "query": (
                "ฉันควรกินอาหารอะไร"
            ),

            "elements": [
                "water",
                "earth",
            ],
        },

        {
            "name": "Mixed Element Menu",

            "query": (
                "มีเมนูอะไรแนะนำบ้าง"
            ),

            "elements": [
                "water",
                "earth",
            ],
        },
    ]


    for case in test_cases:

        print()

        print(
            "=" * 70
        )

        print(
            f"CASE: "
            f"{case['name']}"
        )

        print(
            f"QUERY: "
            f"{case['query']}"
        )

        print(
            "ACTIVE ELEMENTS: "
            + ", ".join(
                case[
                    "elements"
                ]
            )
        )

        intents = (
            detect_query_intents(
                case[
                    "query"
                ]
            )
        )

        print(
            "QUERY INTENTS: "
            + (
                ", ".join(
                    intents
                )
                if intents
                else "-"
            )
        )

        print(
            "=" * 70
        )


        results = retrieve_knowledge(
            query=case[
                "query"
            ],
            active_elements=case[
                "elements"
            ],
            top_k=5,
        )


        for index, item in enumerate(
            results,
            start=1,
        ):

            print(
                f"{index}. "
                f"{item['source_file']} "
                f"| element="
                f"{item['element']} "
                f"| type="
                f"{item['knowledge_type']} "
                f"| score="
                f"{item['retrieval_score']}"
            )