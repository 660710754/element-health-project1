from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from food_retrieval import retrieve_foods


# =========================================================
# Configuration
# =========================================================

TOP_K = 4

BASE_DIR = Path(__file__).resolve().parent

FOODS_PATH = (
    BASE_DIR
    / "data"
    / "foods.csv"
)

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "data"
    / "ir_ground_truth.csv"
)


# =========================================================
# CSV Loader
# =========================================================

def load_csv(
    path: Path
) -> list[dict[str, str]]:
    """
    โหลด CSV และคืนค่าเป็น list[dict]
    """

    if not path.exists():

        raise FileNotFoundError(
            f"ไม่พบไฟล์: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )

    if not rows:

        raise ValueError(
            f"ไฟล์ไม่มีข้อมูล: {path}"
        )

    return rows


# =========================================================
# Load Food Database
# =========================================================

def load_food_database() -> list[dict[str, str]]:
    """
    โหลด foods.csv
    สำหรับตรวจสอบ Ground Truth
    """

    return load_csv(
        FOODS_PATH
    )


# =========================================================
# Load Ground Truth
# =========================================================

def load_ground_truth() -> list[dict[str, Any]]:
    """
    โหลด Ground Truth รูปแบบใหม่

    Columns:

    query_id
    user_input
    relevant_elements
    expected_category
    expected_tastes
    expected_food_ids
    """

    rows = load_csv(
        GROUND_TRUTH_PATH
    )

    required_columns = {
        "query_id",
        "user_input",
        "relevant_elements",
        "expected_category",
        "expected_tastes",
        "expected_food_ids",
    }

    actual_columns = set(
        rows[0].keys()
    )

    missing_columns = (
        required_columns
        -
        actual_columns
    )

    if missing_columns:

        raise ValueError(
            "ir_ground_truth.csv "
            "ขาดคอลัมน์: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    test_cases = []

    for row in rows:

        query_id = (
            row.get(
                "query_id",
                ""
            )
            .strip()
        )

        user_input = (
            row.get(
                "user_input",
                ""
            )
            .strip()
        )

        elements = [
            element.strip().lower()

            for element
            in row.get(
                "relevant_elements",
                ""
            ).split("|")

            if element.strip()
        ]

        expected_category = (
            row.get(
                "expected_category",
                ""
            )
            .strip()
        )

        expected_tastes = [
            taste.strip()

            for taste
            in row.get(
                "expected_tastes",
                ""
            ).split("|")

            if taste.strip()
        ]

        expected_food_ids = {
            food_id.strip()

            for food_id
            in row.get(
                "expected_food_ids",
                ""
            ).split("|")

            if food_id.strip()
        }

        if not query_id:

            raise ValueError(
                "พบ Ground Truth "
                "ที่ไม่มี query_id"
            )

        if not user_input:

            raise ValueError(
                f"{query_id}: "
                "ไม่มี user_input"
            )

        if not elements:

            raise ValueError(
                f"{query_id}: "
                "ไม่มี relevant_elements"
            )

        if not expected_food_ids:

            raise ValueError(
                f"{query_id}: "
                "ไม่มี expected_food_ids"
            )

        test_cases.append(
            {
                "query_id": query_id,
                "user_input": user_input,
                "elements": elements,
                "expected_category": (
                    expected_category
                ),
                "expected_tastes": (
                    expected_tastes
                ),
                "expected_food_ids": (
                    expected_food_ids
                ),
            }
        )

    return test_cases


# =========================================================
# Validate Ground Truth
# =========================================================

def validate_ground_truth(
    foods: list[dict[str, str]],
    test_cases: list[dict[str, Any]],
) -> None:
    """
    ตรวจว่า food_id ใน Ground Truth
    มีอยู่จริงใน foods.csv

    และต้องเป็น recommendation_status
    = recommended
    """

    food_by_id = {}

    for food in foods:

        food_id = (
            food.get(
                "food_id",
                ""
            )
            .strip()
        )

        if food_id:

            food_by_id[
                food_id
            ] = food

    errors = []

    for case in test_cases:

        query_id = case[
            "query_id"
        ]

        expected_food_ids = case[
            "expected_food_ids"
        ]

        for food_id in expected_food_ids:

            if food_id not in food_by_id:

                errors.append(
                    f"{query_id}: "
                    f"ไม่พบ {food_id} "
                    "ใน foods.csv"
                )

                continue

            food = food_by_id[
                food_id
            ]

            status = (
                food.get(
                    "recommendation_status",
                    ""
                )
                .strip()
                .lower()
            )

            if (
                status
                !=
                "recommended"
            ):

                errors.append(
                    f"{query_id}: "
                    f"{food_id} "
                    "ไม่ใช่ recommended"
                )

    if errors:

        error_text = "\n".join(
            errors
        )

        raise ValueError(
            "Ground Truth Validation "
            "ไม่ผ่าน:\n"
            f"{error_text}"
        )


# =========================================================
# Helper
# =========================================================

def get_result_food_id(
    item: dict[str, Any]
) -> str:

    return (
        str(
            item.get(
                "food_id",
                ""
            )
        )
        .strip()
    )


def get_retrieved_ids(
    results: list[dict[str, Any]],
    k: int,
) -> list[str]:
    """
    ดึง Food IDs จาก Top-K
    """

    ids = []

    for item in results[:k]:

        food_id = (
            get_result_food_id(
                item
            )
        )

        if food_id:

            ids.append(
                food_id
            )

    return ids


# =========================================================
# Precision@K
# =========================================================

def calculate_precision_at_k(
    results: list[dict[str, Any]],
    relevant_food_ids: set[str],
    k: int,
) -> float:
    """
    Precision@K

    Precision@K =
    จำนวน Relevant Items ใน Top K
    ------------------------------
                 K
    """

    if k <= 0:
        return 0.0

    retrieved_ids = (
        get_retrieved_ids(
            results,
            k
        )
    )

    relevant_hits = sum(
        1
        for food_id
        in retrieved_ids
        if food_id in relevant_food_ids
    )

    return (
        relevant_hits
        /
        k
    )


# =========================================================
# Recall@K
# =========================================================

def calculate_recall_at_k(
    results: list[dict[str, Any]],
    relevant_food_ids: set[str],
    k: int,
) -> float:
    """
    Recall@K

    Recall@K =
    จำนวน Relevant Items ที่ค้นพบ
    -----------------------------
    จำนวน Relevant Items ทั้งหมด
    """

    if not relevant_food_ids:

        return 0.0

    retrieved_ids = set(
        get_retrieved_ids(
            results,
            k
        )
    )

    relevant_hits = (
        retrieved_ids
        &
        relevant_food_ids
    )

    return (
        len(relevant_hits)
        /
        len(relevant_food_ids)
    )


# =========================================================
# Average Precision@K
# =========================================================

def calculate_average_precision_at_k(
    results: list[dict[str, Any]],
    relevant_food_ids: set[str],
    k: int,
) -> float:
    """
    AP@K

    ประเมินทั้ง:
    - ความถูกต้อง
    - ตำแหน่งของ Relevant Items

    Relevant item ที่อยู่ลำดับสูง
    จะได้คะแนนมากกว่า
    """

    if not relevant_food_ids:

        return 0.0

    hit_count = 0

    precision_sum = 0.0

    seen_ids = set()

    for rank, item in enumerate(
        results[:k],
        start=1
    ):

        food_id = (
            get_result_food_id(
                item
            )
        )

        if not food_id:

            continue

        if food_id in seen_ids:

            continue

        seen_ids.add(
            food_id
        )

        if (
            food_id
            in relevant_food_ids
        ):

            hit_count += 1

            precision_at_rank = (
                hit_count
                /
                rank
            )

            precision_sum += (
                precision_at_rank
            )

    denominator = min(
        len(
            relevant_food_ids
        ),
        k
    )

    if denominator == 0:

        return 0.0

    return (
        precision_sum
        /
        denominator
    )


# =========================================================
# Top-K Accuracy
# =========================================================

def calculate_top_k_accuracy(
    results: list[dict[str, Any]],
    relevant_food_ids: set[str],
    k: int,
) -> float:
    """
    Top-K Accuracy

    ถ้า Top-K มี Relevant Item
    อย่างน้อย 1 ตัว = 1

    ถ้าไม่มีเลย = 0
    """

    retrieved_ids = (
        get_retrieved_ids(
            results,
            k
        )
    )

    for food_id in retrieved_ids:

        if (
            food_id
            in relevant_food_ids
        ):

            return 1.0

    return 0.0


# =========================================================
# Element Coverage@K
# =========================================================

def calculate_element_coverage_at_k(
    results: list[dict[str, Any]],
    expected_elements: list[str],
    k: int,
) -> float:
    """
    Element Coverage@K

    ใช้เป็น Metric เสริม
    โดยเฉพาะ Mixed Element

    ตัวอย่าง:

    expected:
    fire | water

    ถ้า Top-K มี:
    fire + water

    coverage = 1.0

    ถ้ามีแค่ fire

    coverage = 0.5
    """

    if not expected_elements:

        return 0.0

    expected_set = set(
        expected_elements
    )

    found_elements = set()

    for item in results[:k]:

        element = (
            str(
                item.get(
                    "recommended_element",
                    ""
                )
            )
            .strip()
            .lower()
        )

        if (
            element
            in expected_set
        ):

            found_elements.add(
                element
            )

    return (
        len(found_elements)
        /
        len(expected_set)
    )


# =========================================================
# Category Match
# =========================================================

def calculate_category_match(
    results: list[dict[str, Any]],
    expected_category: str,
    k: int,
) -> float:
    """
    ตรวจสัดส่วนผลลัพธ์ที่ตรง Category

    ถ้า Ground Truth ไม่ระบุ Category
    คืนค่า 1.0
    """

    if not expected_category:

        return 1.0

    top_results = (
        results[:k]
    )

    if not top_results:

        return 0.0

    correct = 0

    for item in top_results:

        category = (
            str(
                item.get(
                    "category",
                    ""
                )
            )
            .strip()
        )

        if (
            category
            ==
            expected_category
        ):

            correct += 1

    return (
        correct
        /
        len(top_results)
    )


# =========================================================
# Taste Match
# =========================================================

def calculate_taste_match(
    results: list[dict[str, Any]],
    expected_tastes: list[str],
    k: int,
) -> float:
    """
    ตรวจสัดส่วนผลลัพธ์ที่มีรสชาติ
    ตรงกับ Ground Truth

    ถ้าไม่ได้ระบุ Taste
    คืนค่า 1.0
    """

    if not expected_tastes:

        return 1.0

    top_results = (
        results[:k]
    )

    if not top_results:

        return 0.0

    correct = 0

    for item in top_results:

        profile = (
            str(
                item.get(
                    "food_taste_profile",
                    ""
                )
            )
            .strip()
        )

        if any(
            taste in profile
            for taste
            in expected_tastes
        ):

            correct += 1

    return (
        correct
        /
        len(top_results)
    )


# =========================================================
# Display
# =========================================================

def display_results(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    precision: float,
    recall: float,
    average_precision: float,
    accuracy: float,
    element_coverage: float,
    category_match: float,
    taste_match: float,
) -> None:

    query_id = case[
        "query_id"
    ]

    user_input = case[
        "user_input"
    ]

    elements = case[
        "elements"
    ]

    expected_category = case[
        "expected_category"
    ]

    expected_tastes = case[
        "expected_tastes"
    ]

    relevant_food_ids = case[
        "expected_food_ids"
    ]

    print(
        "=" * 85
    )

    print(
        f"QUERY ID       : "
        f"{query_id}"
    )

    print(
        f"USER INPUT     : "
        f"{user_input}"
    )

    print(
        "ELEMENTS       : "
        + " | ".join(
            elements
        )
    )

    print(
        "CATEGORY       : "
        + (
            expected_category
            if expected_category
            else "-"
        )
    )

    print(
        "TASTE          : "
        + (
            " | ".join(
                expected_tastes
            )
            if expected_tastes
            else "-"
        )
    )

    print(
        f"GROUND TRUTH   : "
        f"{len(relevant_food_ids)} "
        "relevant foods"
    )

    print(
        "=" * 85
    )

    if not results:

        print(
            "ไม่พบผลลัพธ์"
        )

    else:

        for index, item in enumerate(
            results,
            start=1
        ):

            food_id = (
                get_result_food_id(
                    item
                )
            )

            food_name = (
                item.get(
                    "food_name_th",
                    ""
                )
            )

            element = (
                item.get(
                    "recommended_element",
                    ""
                )
            )

            category = (
                item.get(
                    "category",
                    ""
                )
            )

            taste = (
                item.get(
                    "food_taste_profile",
                    ""
                )
            )

            score = (
                item.get(
                    "ir_score",
                    0.0
                )
            )

            is_relevant = (
                food_id
                in relevant_food_ids
            )

            relevance_mark = (
                "✓ Relevant"
                if is_relevant
                else
                "✗ Not Relevant"
            )

            print(
                f"{index}. "
                f"[{food_id}] "
                f"{food_name} "
                f"| element={element} "
                f"| category={category} "
                f"| taste={taste} "
                f"| score={score} "
                f"| {relevance_mark}"
            )

    print()

    print(
        f"Precision@{TOP_K}: "
        f"{precision:.3f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{recall:.3f}"
    )

    print(
        f"AP@{TOP_K}: "
        f"{average_precision:.3f}"
    )

    print(
        f"Top-K Accuracy: "
        f"{accuracy:.3f}"
    )

    print(
        f"Element Coverage@{TOP_K}: "
        f"{element_coverage:.3f}"
    )

    print(
        f"Category Match@{TOP_K}: "
        f"{category_match:.3f}"
    )

    print(
        f"Taste Match@{TOP_K}: "
        f"{taste_match:.3f}"
    )

    print()


# =========================================================
# Main Evaluation
# =========================================================

def run_ir_evaluation() -> None:

    print()

    print(
        "เริ่มประเมิน "
        "Hybrid IR Recommendation System"
    )

    print(
        "Query-Level Ground Truth Evaluation"
    )

    print()


    # =====================================================
    # Load Data
    # =====================================================

    foods = (
        load_food_database()
    )

    test_cases = (
        load_ground_truth()
    )


    print(
        f"โหลด foods.csv: "
        f"{len(foods)} rows"
    )

    print(
        f"โหลด Ground Truth: "
        f"{len(test_cases)} queries"
    )

    print()


    # =====================================================
    # Validate Ground Truth
    # =====================================================

    validate_ground_truth(
        foods=foods,
        test_cases=test_cases,
    )


    print(
        "✅ Ground Truth Validation ผ่าน"
    )

    print()


    # =====================================================
    # Score Lists
    # =====================================================

    precision_scores = []

    recall_scores = []

    ap_scores = []

    accuracy_scores = []

    coverage_scores = []

    category_scores = []

    taste_scores = []


    # =====================================================
    # Evaluate Each Query
    # =====================================================

    for case in test_cases:

        query = case[
            "user_input"
        ]

        elements = case[
            "elements"
        ]

        relevant_food_ids = case[
            "expected_food_ids"
        ]


        # -------------------------------------------------
        # IMPORTANT
        #
        # ส่ง User Input จริงเข้า IR
        # -------------------------------------------------

        results = (
            retrieve_foods(
                query=query,
                active_elements=elements,
                top_k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Precision@K
        # -------------------------------------------------

        precision = (
            calculate_precision_at_k(
                results=results,
                relevant_food_ids=(
                    relevant_food_ids
                ),
                k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Recall@K
        # -------------------------------------------------

        recall = (
            calculate_recall_at_k(
                results=results,
                relevant_food_ids=(
                    relevant_food_ids
                ),
                k=TOP_K,
            )
        )


        # -------------------------------------------------
        # AP@K
        # -------------------------------------------------

        average_precision = (
            calculate_average_precision_at_k(
                results=results,
                relevant_food_ids=(
                    relevant_food_ids
                ),
                k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Top-K Accuracy
        # -------------------------------------------------

        accuracy = (
            calculate_top_k_accuracy(
                results=results,
                relevant_food_ids=(
                    relevant_food_ids
                ),
                k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Element Coverage
        # -------------------------------------------------

        element_coverage = (
            calculate_element_coverage_at_k(
                results=results,
                expected_elements=elements,
                k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Category Match
        # -------------------------------------------------

        category_match = (
            calculate_category_match(
                results=results,
                expected_category=case[
                    "expected_category"
                ],
                k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Taste Match
        # -------------------------------------------------

        taste_match = (
            calculate_taste_match(
                results=results,
                expected_tastes=case[
                    "expected_tastes"
                ],
                k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Save Scores
        # -------------------------------------------------

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        ap_scores.append(
            average_precision
        )

        accuracy_scores.append(
            accuracy
        )

        coverage_scores.append(
            element_coverage
        )

        category_scores.append(
            category_match
        )

        taste_scores.append(
            taste_match
        )


        # -------------------------------------------------
        # Display Query Result
        # -------------------------------------------------

        display_results(
            case=case,
            results=results,
            precision=precision,
            recall=recall,
            average_precision=(
                average_precision
            ),
            accuracy=accuracy,
            element_coverage=(
                element_coverage
            ),
            category_match=(
                category_match
            ),
            taste_match=(
                taste_match
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


    mean_precision = (
        sum(
            precision_scores
        )
        /
        total_cases
    )

    mean_recall = (
        sum(
            recall_scores
        )
        /
        total_cases
    )

    mean_average_precision = (
        sum(
            ap_scores
        )
        /
        total_cases
    )

    mean_accuracy = (
        sum(
            accuracy_scores
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

    mean_category_match = (
        sum(
            category_scores
        )
        /
        total_cases
    )

    mean_taste_match = (
        sum(
            taste_scores
        )
        /
        total_cases
    )


    # =====================================================
    # Final Report
    # =====================================================

    print(
        "=" * 85
    )

    print(
        "HYBRID IR EVALUATION SUMMARY"
    )

    print(
        "=" * 85
    )

    print(
        f"Queries evaluated: "
        f"{total_cases}"
    )

    print()

    print(
        f"Mean Precision@{TOP_K}: "
        f"{mean_precision:.3f}"
    )

    print(
        f"Mean Recall@{TOP_K}: "
        f"{mean_recall:.3f}"
    )

    print(
        f"MAP@{TOP_K}: "
        f"{mean_average_precision:.3f}"
    )

    print(
        "Average Top-K Accuracy: "
        f"{mean_accuracy:.3f}"
    )

    print()

    print(
        f"Mean Element Coverage@{TOP_K}: "
        f"{mean_coverage:.3f}"
    )

    print(
        f"Mean Category Match@{TOP_K}: "
        f"{mean_category_match:.3f}"
    )

    print(
        f"Mean Taste Match@{TOP_K}: "
        f"{mean_taste_match:.3f}"
    )

    print(
        "=" * 85
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    run_ir_evaluation()