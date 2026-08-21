from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from rag import (
    RAGSystem,
    determine_active_elements,
)

from knowledge_retrieval import (
    detect_query_intents,
)


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "data"
    / "rag_ground_truth.csv"
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

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )

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
    แปลงค่าแบบ:

    water|earth

    เป็น:

    ["water", "earth"]
    """

    if not value:
        return []

    return [
        item.strip()
        for item
        in value.split("|")
        if item.strip()
    ]


# =========================================================
# Load RAG Ground Truth
# =========================================================

def load_ground_truth() -> list[
    dict[str, Any]
]:
    """
    โหลด Ground Truth
    จาก data/rag_ground_truth.csv
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

                "expected_elements": (
                    parse_pipe_values(
                        row[
                            "expected_elements"
                        ]
                    )
                ),

                "expected_intents": (
                    parse_pipe_values(
                        row[
                            "expected_intents"
                        ]
                    )
                ),

                "expected_source_types": (
                    parse_pipe_values(
                        row[
                            "expected_source_types"
                        ]
                    )
                ),
            }
        )


    return cases


# =========================================================
# Metric 1
# Element Accuracy
# =========================================================

def calculate_element_accuracy(
    actual_elements: list[str],
    expected_elements: list[str],
) -> float:
    """
    Element Accuracy

    1.0 = active elements ตรงกับ Ground Truth ทุกตัว
    0.0 = ไม่ตรง
    """

    actual_set = set(
        actual_elements
    )

    expected_set = set(
        expected_elements
    )

    if actual_set == expected_set:
        return 1.0

    return 0.0


# =========================================================
# Metric 2
# Intent Accuracy
# =========================================================

def calculate_intent_accuracy(
    actual_intents: list[str],
    expected_intents: list[str],
) -> float:
    """
    Intent Accuracy

    1.0 = intent ตรงกับ Ground Truth ทุกตัว
    0.0 = ไม่ตรง
    """

    actual_set = set(
        actual_intents
    )

    expected_set = set(
        expected_intents
    )

    if actual_set == expected_set:
        return 1.0

    return 0.0


# =========================================================
# Metric 3
# Source-Type Precision@K
# =========================================================

def calculate_source_type_precision_at_k(
    results: list[dict[str, Any]],
    expected_source_types: list[str],
    k: int,
) -> float:
    """
    Source-Type Precision@K

    ดูว่า Top-K Retrieved Chunks
    อยู่ใน knowledge type ที่คาดหวังกี่รายการ

    ตัวอย่าง:

    Expected:
        food

    Top 5:
        profile
        food
        food
        food
        food

    Precision@5:
        4/5 = 0.8
    """

    top_results = (
        results[:k]
    )

    if not top_results:
        return 0.0


    expected_set = set(
        expected_source_types
    )


    correct = 0


    for item in top_results:

        knowledge_type = str(
            item.get(
                "knowledge_type",
                "",
            )
        ).strip()


        if (
            knowledge_type
            in expected_set
        ):

            correct += 1


    return (
        correct
        /
        len(top_results)
    )


# =========================================================
# Metric 4
# Element Coverage@K
# =========================================================

def calculate_element_coverage_at_k(
    results: list[dict[str, Any]],
    expected_elements: list[str],
    k: int,
) -> float:
    """
    Element Coverage@K

    ใช้ดูว่า Top-K
    ครอบคลุมธาตุที่ต้องการครบหรือไม่

    ตัวอย่าง:

    Expected:
        water + earth

    Retrieved:
        water
        water
        earth
        water
        earth

    Coverage:
        2/2 = 1.0
    """

    if not expected_elements:
        return 0.0


    expected_set = set(
        expected_elements
    )


    found_elements = {
        str(
            item.get(
                "element",
                "",
            )
        ).strip()

        for item in results[:k]

        if str(
            item.get(
                "element",
                "",
            )
        ).strip()
        in expected_set
    }


    return (
        len(found_elements)
        /
        len(expected_set)
    )


# =========================================================
# Metric 5
# Expected-Type Hit@K
# =========================================================

def calculate_expected_type_hit_at_k(
    results: list[dict[str, Any]],
    expected_source_types: list[str],
    k: int,
) -> float:
    """
    Expected-Type Hit@K

    ถ้า Top-K มี knowledge type
    ที่คาดหวังอย่างน้อย 1 รายการ = 1
    """

    expected_set = set(
        expected_source_types
    )


    for item in results[:k]:

        knowledge_type = str(
            item.get(
                "knowledge_type",
                "",
            )
        ).strip()


        if (
            knowledge_type
            in expected_set
        ):

            return 1.0


    return 0.0


# =========================================================
# Metric 6
# Top-1 Type Accuracy
# =========================================================

def calculate_top1_type_accuracy(
    results: list[dict[str, Any]],
    expected_source_types: list[str],
) -> float:
    """
    Top-1 Type Accuracy

    เช็กว่าอันดับ 1
    เป็น knowledge type ที่ตรงกับ Ground Truth หรือไม่
    """

    if not results:
        return 0.0


    expected_set = set(
        expected_source_types
    )


    top_type = str(
        results[0].get(
            "knowledge_type",
            "",
        )
    ).strip()


    if top_type in expected_set:
        return 1.0


    return 0.0


# =========================================================
# Display Single Case
# =========================================================

def display_case_result(
    case: dict[str, Any],
    actual_elements: list[str],
    actual_intents: list[str],
    results: list[dict[str, Any]],
    element_accuracy: float,
    intent_accuracy: float,
    source_precision: float,
    coverage: float,
    type_hit: float,
    top1_type_accuracy: float,
) -> None:
    """
    แสดงผล Evaluation
    ของแต่ละ Query
    """

    print(
        "=" * 78
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
        "EXPECTED ELEMENTS : "
        + (
            ", ".join(
                case[
                    "expected_elements"
                ]
            )
            or "-"
        )
    )

    print(
        "ACTUAL ELEMENTS   : "
        + (
            ", ".join(
                actual_elements
            )
            or "-"
        )
    )

    print(
        "EXPECTED INTENTS  : "
        + (
            ", ".join(
                case[
                    "expected_intents"
                ]
            )
            or "-"
        )
    )

    print(
        "ACTUAL INTENTS    : "
        + (
            ", ".join(
                actual_intents
            )
            or "-"
        )
    )

    print(
        "EXPECTED TYPES    : "
        + (
            ", ".join(
                case[
                    "expected_source_types"
                ]
            )
            or "-"
        )
    )

    print(
        "=" * 78
    )


    if not results:

        print(
            "ไม่พบ Retrieval Results"
        )

    else:

        for index, item in enumerate(
            results,
            start=1,
        ):

            print(
                f"{index}. "
                f"{item.get('source_file')} "
                f"| element="
                f"{item.get('element')} "
                f"| type="
                f"{item.get('knowledge_type')} "
                f"| score="
                f"{item.get('retrieval_score')}"
            )


    print()


    print(
        "Element Accuracy: "
        f"{element_accuracy:.3f}"
    )

    print(
        "Intent Accuracy: "
        f"{intent_accuracy:.3f}"
    )

    print(
        f"Source-Type Precision@{TOP_K}: "
        f"{source_precision:.3f}"
    )

    print(
        f"Element Coverage@{TOP_K}: "
        f"{coverage:.3f}"
    )

    print(
        f"Expected-Type Hit@{TOP_K}: "
        f"{type_hit:.3f}"
    )

    print(
        "Top-1 Type Accuracy: "
        f"{top1_type_accuracy:.3f}"
    )

    print()


# =========================================================
# Main Evaluation
# =========================================================

def run_rag_retrieval_evaluation() -> None:
    """
    ประเมิน Retrieval Layer ของ RAG

    หมายเหตุ:
    ฟังก์ชันนี้ไม่เรียก Ollama
    จึงรันได้เร็วกว่า Full RAG Evaluation
    """

    print()

    print(
        "เริ่มประเมิน RAG Retrieval System"
    )

    print()


    # -----------------------------------------------------
    # Load Ground Truth
    # -----------------------------------------------------

    test_cases = (
        load_ground_truth()
    )


    print(
        f"โหลด RAG Ground Truth: "
        f"{len(test_cases)} queries"
    )

    print()


    # -----------------------------------------------------
    # RAG System
    # -----------------------------------------------------

    rag = RAGSystem(
        top_k=TOP_K
    )


    # -----------------------------------------------------
    # Score Lists
    # -----------------------------------------------------

    element_scores: list[
        float
    ] = []

    intent_scores: list[
        float
    ] = []

    source_precision_scores: list[
        float
    ] = []

    coverage_scores: list[
        float
    ] = []

    type_hit_scores: list[
        float
    ] = []

    top1_type_scores: list[
        float
    ] = []


    # -----------------------------------------------------
    # Evaluate Queries
    # -----------------------------------------------------

    for case in test_cases:

        query = case[
            "query"
        ]

        primary_element = case[
            "primary_element"
        ]

        secondary_element = case[
            "secondary_element"
        ]


        # ---------------------------------------------
        # Active Elements
        # ---------------------------------------------

        actual_elements = (
            determine_active_elements(
                question=query,
                primary_element=(
                    primary_element
                ),
                secondary_element=(
                    secondary_element
                ),
            )
        )


        # ---------------------------------------------
        # Query Intents
        # ---------------------------------------------

        actual_intents = (
            detect_query_intents(
                query
            )
        )


        # ---------------------------------------------
        # Retrieval
        # ---------------------------------------------

        if actual_elements:

            results = rag.retrieve(
                question=query,
                active_elements=(
                    actual_elements
                ),
            )

        else:

            results = []


        # ---------------------------------------------
        # Calculate Metrics
        # ---------------------------------------------

        element_accuracy = (
            calculate_element_accuracy(
                actual_elements=(
                    actual_elements
                ),
                expected_elements=(
                    case[
                        "expected_elements"
                    ]
                ),
            )
        )


        intent_accuracy = (
            calculate_intent_accuracy(
                actual_intents=(
                    actual_intents
                ),
                expected_intents=(
                    case[
                        "expected_intents"
                    ]
                ),
            )
        )


        source_precision = (
            calculate_source_type_precision_at_k(
                results=results,
                expected_source_types=(
                    case[
                        "expected_source_types"
                    ]
                ),
                k=TOP_K,
            )
        )


        coverage = (
            calculate_element_coverage_at_k(
                results=results,
                expected_elements=(
                    case[
                        "expected_elements"
                    ]
                ),
                k=TOP_K,
            )
        )


        type_hit = (
            calculate_expected_type_hit_at_k(
                results=results,
                expected_source_types=(
                    case[
                        "expected_source_types"
                    ]
                ),
                k=TOP_K,
            )
        )


        top1_type_accuracy = (
            calculate_top1_type_accuracy(
                results=results,
                expected_source_types=(
                    case[
                        "expected_source_types"
                    ]
                ),
            )
        )


        # ---------------------------------------------
        # Save Scores
        # ---------------------------------------------

        element_scores.append(
            element_accuracy
        )

        intent_scores.append(
            intent_accuracy
        )

        source_precision_scores.append(
            source_precision
        )

        coverage_scores.append(
            coverage
        )

        type_hit_scores.append(
            type_hit
        )

        top1_type_scores.append(
            top1_type_accuracy
        )


        # ---------------------------------------------
        # Display
        # ---------------------------------------------

        display_case_result(
            case=case,
            actual_elements=(
                actual_elements
            ),
            actual_intents=(
                actual_intents
            ),
            results=results,
            element_accuracy=(
                element_accuracy
            ),
            intent_accuracy=(
                intent_accuracy
            ),
            source_precision=(
                source_precision
            ),
            coverage=coverage,
            type_hit=type_hit,
            top1_type_accuracy=(
                top1_type_accuracy
            ),
        )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total_cases = len(
        test_cases
    )


    if total_cases == 0:

        print(
            "ไม่มี Test Case"
        )

        return


    mean_element_accuracy = (
        sum(
            element_scores
        )
        /
        total_cases
    )


    mean_intent_accuracy = (
        sum(
            intent_scores
        )
        /
        total_cases
    )


    mean_source_precision = (
        sum(
            source_precision_scores
        )
        /
        total_cases
    )


    mean_coverage = (
        sum(
            coverage_scores
        )
        /
        total_cases
    )


    mean_type_hit = (
        sum(
            type_hit_scores
        )
        /
        total_cases
    )


    mean_top1_type_accuracy = (
        sum(
            top1_type_scores
        )
        /
        total_cases
    )


    print(
        "=" * 78
    )

    print(
        "RAG Retrieval Evaluation Summary"
    )

    print(
        "=" * 78
    )


    print(
        f"Queries Evaluated: "
        f"{total_cases}"
    )


    print(
        "Element Accuracy: "
        f"{mean_element_accuracy:.3f}"
    )


    print(
        "Intent Accuracy: "
        f"{mean_intent_accuracy:.3f}"
    )


    print(
        f"Mean Source-Type "
        f"Precision@{TOP_K}: "
        f"{mean_source_precision:.3f}"
    )


    print(
        f"Mean Element "
        f"Coverage@{TOP_K}: "
        f"{mean_coverage:.3f}"
    )


    print(
        f"Expected-Type "
        f"Hit@{TOP_K}: "
        f"{mean_type_hit:.3f}"
    )


    print(
        "Top-1 Type Accuracy: "
        f"{mean_top1_type_accuracy:.3f}"
    )


    print(
        "=" * 78
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    run_rag_retrieval_evaluation()