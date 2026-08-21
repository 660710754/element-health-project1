from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from rag import RAGSystem


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "data"
    / "rag_generation_ground_truth.csv"
)

MODEL_NAME = (
    "scb10x/typhoon2.5-qwen3-4b:latest"
)

TOP_K = 5


# =========================================================
# CSV Loader
# =========================================================

def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    """
    โหลดไฟล์ CSV
    """

    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"ไฟล์ {path.name} ไม่มีข้อมูล"
        )

    return rows


# =========================================================
# Parse Pipe-Separated Values
# =========================================================

def parse_pipe_values(
    value: str,
) -> list[str]:
    """
    แปลง:

    a|b|c

    เป็น:

    ["a", "b", "c"]
    """

    if not value:
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


# =========================================================
# Load Ground Truth
# =========================================================

def load_ground_truth() -> list[
    dict[str, Any]
]:
    """
    โหลด Ground Truth
    สำหรับ RAG Generation Evaluation
    """

    rows = load_csv(
        GROUND_TRUTH_PATH
    )

    cases: list[
        dict[str, Any]
    ] = []

    for row in rows:

        cases.append(
            {
                "query_id": (
                    row[
                        "query_id"
                    ].strip()
                ),

                "query": (
                    row[
                        "query"
                    ].strip()
                ),

                "primary_element": (
                    row[
                        "primary_element"
                    ].strip()
                    or None
                ),

                "secondary_element": (
                    row[
                        "secondary_element"
                    ].strip()
                    or None
                ),

                "expected_keywords": (
                    parse_pipe_values(
                        row[
                            "expected_keywords"
                        ]
                    )
                ),

                "forbidden_keywords": (
                    parse_pipe_values(
                        row[
                            "forbidden_keywords"
                        ]
                    )
                ),
            }
        )

    return cases


# =========================================================
# Normalize Text
# =========================================================

def normalize_text(
    text: str,
) -> str:
    """
    Normalize ข้อความสำหรับการเทียบแบบง่าย
    """

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# =========================================================
# Keyword Coverage
# =========================================================

def calculate_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """
    วัดว่า expected keywords
    ปรากฏในคำตอบกี่สัดส่วน

    ตัวอย่าง:
    expected = 4 คำ
    พบ 3 คำ
    score = 0.75
    """

    if not expected_keywords:
        return 1.0

    answer_n = normalize_text(
        answer
    )

    found = 0

    for keyword in expected_keywords:

        keyword_n = normalize_text(
            keyword
        )

        if keyword_n in answer_n:
            found += 1

    return (
        found
        /
        len(expected_keywords)
    )


# =========================================================
# Forbidden Content Pass
# =========================================================

def calculate_forbidden_content_pass(
    answer: str,
    forbidden_keywords: list[str],
) -> float:
    """
    1.0 = ไม่พบ forbidden keyword
    0.0 = พบอย่างน้อย 1 keyword
    """

    if not forbidden_keywords:
        return 1.0

    answer_n = normalize_text(
        answer
    )

    for keyword in forbidden_keywords:

        keyword_n = normalize_text(
            keyword
        )

        if keyword_n in answer_n:
            return 0.0

    return 1.0


# =========================================================
# Build Context Text from Retrieved Chunks
# =========================================================

def build_context_text(
    retrieved_chunks: list[
        dict[str, Any]
    ],
) -> str:
    """
    รวม text ของ retrieved chunks
    เพื่อใช้ประเมิน groundedness แบบ heuristic
    """

    parts: list[str] = []

    for item in retrieved_chunks:

        text = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        if text:
            parts.append(
                text
            )

    return "\n".join(
        parts
    )


# =========================================================
# Thai/Word Tokenization - Simple Heuristic
# =========================================================

def extract_meaningful_tokens(
    text: str,
) -> list[str]:
    """
    Token heuristic แบบง่าย

    ไม่ใช้ tokenizer ภาษาไทยเพิ่มเติม
    เพื่อให้ dependency ต่ำ

    ตัด:
    - punctuation
    - token สั้นเกินไป
    """

    cleaned = re.sub(
        r"[^\wก-๙]+",
        " ",
        text.lower(),
    )

    tokens = [
        token.strip()
        for token in cleaned.split()
        if len(token.strip()) >= 2
    ]

    return tokens


# =========================================================
# Context Groundedness Heuristic
# =========================================================

def calculate_context_groundedness(
    answer: str,
    context: str,
) -> float:
    """
    Groundedness แบบ heuristic

    วัดสัดส่วน token สำคัญจากคำตอบ
    ที่พบใน Retrieved Context

    หมายเหตุ:
    ไม่ใช่ semantic groundedness แบบสมบูรณ์
    แต่ใช้เป็น baseline ที่ตรวจสอบย้อนกลับได้
    """

    if not answer.strip():
        return 0.0

    if not context.strip():
        return 0.0

    answer_tokens = set(
        extract_meaningful_tokens(
            answer
        )
    )

    context_tokens = set(
        extract_meaningful_tokens(
            context
        )
    )

    if not answer_tokens:
        return 0.0

    matched_tokens = (
        answer_tokens
        &
        context_tokens
    )

    return (
        len(matched_tokens)
        /
        len(answer_tokens)
    )


# =========================================================
# Source Coverage
# =========================================================

def calculate_source_presence(
    result: dict[str, Any],
) -> float:
    """
    ตรวจว่าคำตอบ RAG มี Retrieved Sources
    ประกอบหรือไม่
    """

    sources = result.get(
        "sources",
        [],
    )

    retrieved_chunks = result.get(
        "retrieved_chunks",
        [],
    )

    if (
        sources
        and retrieved_chunks
    ):
        return 1.0

    return 0.0


# =========================================================
# Insufficient-Information Detection
# =========================================================

def is_no_information_answer(
    answer: str,
) -> bool:
    """
    ตรวจข้อความ fallback
    """

    phrase = (
        "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"
    )

    return phrase in answer


# =========================================================
# Display Case
# =========================================================

def display_case_result(
    case: dict[str, Any],
    result: dict[str, Any],
    keyword_coverage: float,
    groundedness: float,
    forbidden_pass: float,
    source_presence: float,
) -> None:
    """
    แสดงผลแต่ละ Query
    """

    print(
        "=" * 80
    )

    print(
        f"QUERY ID : "
        f"{case['query_id']}"
    )

    print(
        f"QUERY    : "
        f"{case['query']}"
    )

    print(
        "PRIMARY  : "
        f"{case['primary_element'] or '-'}"
    )

    print(
        "SECONDARY: "
        f"{case['secondary_element'] or '-'}"
    )

    print(
        "EXPECTED KEYWORDS: "
        + (
            ", ".join(
                case[
                    "expected_keywords"
                ]
            )
            or "-"
        )
    )

    print(
        "FORBIDDEN KEYWORDS: "
        + (
            ", ".join(
                case[
                    "forbidden_keywords"
                ]
            )
            or "-"
        )
    )

    print(
        "=" * 80
    )

    print()

    print(
        "RAG ANSWER"
    )

    print(
        "-" * 80
    )

    print(
        result.get(
            "answer",
            "",
        )
    )

    print()

    print(
        "SOURCES"
    )

    print(
        "-" * 80
    )

    sources = result.get(
        "sources",
        [],
    )

    if sources:

        for source in sources:

            print(
                f"- {source}"
            )

    else:

        print(
            "ไม่พบ Source"
        )

    print()

    print(
        "METRICS"
    )

    print(
        "-" * 80
    )

    print(
        "Keyword Coverage: "
        f"{keyword_coverage:.3f}"
    )

    print(
        "Context Groundedness: "
        f"{groundedness:.3f}"
    )

    print(
        "Forbidden-Content Pass: "
        f"{forbidden_pass:.3f}"
    )

    print(
        "Source Presence: "
        f"{source_presence:.3f}"
    )

    print(
        "No-Information Answer: "
        f"{is_no_information_answer(
            result.get('answer', '')
        )}"
    )

    print()


# =========================================================
# Main Evaluation
# =========================================================

def run_rag_generation_evaluation() -> None:
    """
    ประเมิน RAG Generation

    NOTE:
    ฟังก์ชันนี้เรียก Ollama จริง
    จึงอาจใช้เวลานาน
    """

    print()

    print(
        "เริ่มประเมิน RAG Generation System"
    )

    print()

    test_cases = (
        load_ground_truth()
    )

    print(
        f"โหลด Generation Ground Truth: "
        f"{len(test_cases)} queries"
    )

    print()


    rag = RAGSystem(
        model=MODEL_NAME,
        top_k=TOP_K,
    )


    keyword_scores: list[
        float
    ] = []

    groundedness_scores: list[
        float
    ] = []

    forbidden_scores: list[
        float
    ] = []

    source_scores: list[
        float
    ] = []


    for case in test_cases:

        # -------------------------------------------------
        # Full RAG
        # -------------------------------------------------

        result = rag.answer(
            question=case[
                "query"
            ],
            primary_element=case[
                "primary_element"
            ],
            secondary_element=case[
                "secondary_element"
            ],
        )


        answer = str(
            result.get(
                "answer",
                "",
            )
        )


        retrieved_chunks = (
            result.get(
                "retrieved_chunks",
                [],
            )
        )


        context_text = (
            build_context_text(
                retrieved_chunks
            )
        )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        keyword_coverage = (
            calculate_keyword_coverage(
                answer=answer,
                expected_keywords=case[
                    "expected_keywords"
                ],
            )
        )


        groundedness = (
            calculate_context_groundedness(
                answer=answer,
                context=context_text,
            )
        )


        forbidden_pass = (
            calculate_forbidden_content_pass(
                answer=answer,
                forbidden_keywords=case[
                    "forbidden_keywords"
                ],
            )
        )


        source_presence = (
            calculate_source_presence(
                result
            )
        )


        keyword_scores.append(
            keyword_coverage
        )

        groundedness_scores.append(
            groundedness
        )

        forbidden_scores.append(
            forbidden_pass
        )

        source_scores.append(
            source_presence
        )


        display_case_result(
            case=case,
            result=result,
            keyword_coverage=(
                keyword_coverage
            ),
            groundedness=(
                groundedness
            ),
            forbidden_pass=(
                forbidden_pass
            ),
            source_presence=(
                source_presence
            ),
        )


    # =====================================================
    # Summary
    # =====================================================

    total_cases = len(
        test_cases
    )

    if total_cases == 0:

        print(
            "ไม่มี Test Case"
        )

        return


    mean_keyword_coverage = (
        sum(
            keyword_scores
        )
        /
        total_cases
    )


    mean_groundedness = (
        sum(
            groundedness_scores
        )
        /
        total_cases
    )


    mean_forbidden_pass = (
        sum(
            forbidden_scores
        )
        /
        total_cases
    )


    mean_source_presence = (
        sum(
            source_scores
        )
        /
        total_cases
    )


    print(
        "=" * 80
    )

    print(
        "RAG Generation Evaluation Summary"
    )

    print(
        "=" * 80
    )

    print(
        f"Queries Evaluated: "
        f"{total_cases}"
    )

    print(
        "Mean Keyword Coverage: "
        f"{mean_keyword_coverage:.3f}"
    )

    print(
        "Mean Context Groundedness: "
        f"{mean_groundedness:.3f}"
    )

    print(
        "Mean Forbidden-Content Pass: "
        f"{mean_forbidden_pass:.3f}"
    )

    print(
        "Mean Source Presence: "
        f"{mean_source_presence:.3f}"
    )

    print(
        "=" * 80
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    run_rag_generation_evaluation()