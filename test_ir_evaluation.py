from __future__ import annotations

import csv
from pathlib import Path

from food_retrieval import retrieve_foods_by_element


# =========================================================
# Configuration
# =========================================================

TOP_K = 5

BASE_DIR = Path(__file__).resolve().parent

FOODS_PATH = BASE_DIR / "data" / "foods.csv"
GROUND_TRUTH_PATH = BASE_DIR / "data" / "ir_ground_truth.csv"


# =========================================================
# CSV Loader
# =========================================================

def load_csv(path: Path) -> list[dict]:
    """
    โหลดไฟล์ CSV และคืนค่าเป็น list[dict]
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

        reader = csv.DictReader(file)

        return list(reader)


# =========================================================
# Load Ground Truth Queries
# =========================================================

def load_ground_truth() -> list[dict]:
    """
    โหลด Query จาก data/ir_ground_truth.csv

    ตัวอย่าง:
    Q001,อาหารธาตุไฟ,fire
    Q005,อาหารธาตุไฟและธาตุน้ำ,fire|water
    """

    rows = load_csv(
        GROUND_TRUTH_PATH
    )

    test_cases = []


    for row in rows:

        elements = [
            element.strip()
            for element
            in row["relevant_elements"].split("|")
            if element.strip()
        ]


        test_cases.append(
            {
                "query_id": row["query_id"].strip(),
                "query": row["query"].strip(),
                "elements": elements,
            }
        )


    return test_cases


# =========================================================
# Build Relevant Food Set
# =========================================================

def build_relevant_food_set(
    foods: list[dict],
    relevant_elements: list[str],
) -> set[str]:
    """
    สร้าง Ground Truth Set ของอาหาร

    relevant =
    recommended_element อยู่ในธาตุที่กำหนด
    และ
    recommendation_status == recommended

    ใช้ set เพื่อไม่ให้นับชื่ออาหารซ้ำ
    """

    relevant_foods: set[str] = set()


    for item in foods:

        element = (
            item.get(
                "recommended_element",
                ""
            )
            .strip()
            .lower()
        )


        status = (
            item.get(
                "recommendation_status",
                ""
            )
            .strip()
            .lower()
        )


        food_name = (
            item.get(
                "food_name_th",
                ""
            )
            .strip()
        )


        if (
            element in relevant_elements
            and status == "recommended"
            and food_name
        ):

            relevant_foods.add(
                food_name
            )


    return relevant_foods


# =========================================================
# Metrics
# =========================================================

def calculate_precision_at_k(
    results: list[dict],
    relevant_foods: set[str],
    k: int,
) -> float:
    """
    Precision@K

    จำนวน Relevant Items ที่พบใน Top K
    หารด้วยจำนวนผลลัพธ์ที่พิจารณา
    """

    top_results = results[:k]


    if not top_results:
        return 0.0


    correct = 0


    for item in top_results:

        food_name = item[
            "food_name_th"
        ].strip()


        if food_name in relevant_foods:
            correct += 1


    return (
        correct
        /
        len(top_results)
    )


def calculate_recall_at_k(
    results: list[dict],
    relevant_foods: set[str],
    k: int,
) -> float:
    """
    Recall@K

    จำนวน Relevant Items ที่ค้นพบใน Top K
    หารด้วย Relevant Items ทั้งหมด
    """

    if not relevant_foods:
        return 0.0


    retrieved_relevant = set()


    for item in results[:k]:

        food_name = item[
            "food_name_th"
        ].strip()


        if food_name in relevant_foods:

            retrieved_relevant.add(
                food_name
            )


    return (
        len(retrieved_relevant)
        /
        len(relevant_foods)
    )


def calculate_average_precision_at_k(
    results: list[dict],
    relevant_foods: set[str],
    k: int,
) -> float:
    """
    Average Precision@K

    ดูว่า Relevant Items ปรากฏอยู่ในอันดับต้น ๆ มากน้อยเพียงใด
    """

    if not relevant_foods:
        return 0.0


    hit_count = 0

    precision_sum = 0.0

    seen_foods = set()


    for rank, item in enumerate(
        results[:k],
        start=1
    ):

        food_name = item[
            "food_name_th"
        ].strip()


        if food_name in seen_foods:
            continue


        seen_foods.add(
            food_name
        )


        if food_name in relevant_foods:

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
        len(relevant_foods),
        k
    )


    if denominator == 0:
        return 0.0


    return (
        precision_sum
        /
        denominator
    )


def calculate_top_k_accuracy(
    results: list[dict],
    relevant_foods: set[str],
    k: int,
) -> float:
    """
    Top-K Accuracy

    ถ้าใน Top K มี Relevant Item
    อย่างน้อย 1 รายการ = 1
    """

    for item in results[:k]:

        food_name = item[
            "food_name_th"
        ].strip()


        if food_name in relevant_foods:
            return 1.0


    return 0.0


def calculate_element_coverage_at_k(
    results: list[dict],
    expected_elements: list[str],
    k: int,
) -> float:
    """
    Element Coverage@K

    ใช้ตรวจกรณีธาตุผสมว่า
    ใน Top K มีผลลัพธ์ครอบคลุมธาตุที่ต้องการครบหรือไม่

    ตัวอย่าง:
    expected = ["fire", "water"]

    ถ้า Top 5 มีทั้ง fire และ water
    coverage = 2/2 = 1.0

    ถ้ามีแต่ water
    coverage = 1/2 = 0.5
    """

    if not expected_elements:
        return 0.0


    expected_set = set(
        expected_elements
    )


    found_elements = set()


    for item in results[:k]:

        element = (
            item.get(
                "recommended_element",
                ""
            )
            .strip()
            .lower()
        )


        if element in expected_set:

            found_elements.add(
                element
            )


    return (
        len(found_elements)
        /
        len(expected_set)
    )


# =========================================================
# Display
# =========================================================

def display_results(
    query_id: str,
    query: str,
    elements: list[str],
    results: list[dict],
    relevant_foods: set[str],
    precision: float,
    recall: float,
    average_precision: float,
    accuracy: float,
    element_coverage: float,
):

    print(
        "=" * 70
    )

    print(
        f"QUERY ID : {query_id}"
    )

    print(
        f"QUERY    : {query}"
    )

    print(
        "ELEMENTS : "
        + ", ".join(elements)
    )

    print(
        f"GROUND TRUTH : "
        f"{len(relevant_foods)} relevant foods"
    )

    print(
        "=" * 70
    )


    if not results:

        print(
            "ไม่พบผลลัพธ์\n"
        )

        return


    for index, item in enumerate(
        results,
        start=1
    ):

        food_name = item[
            "food_name_th"
        ].strip()


        is_relevant = (
            food_name
            in relevant_foods
        )


        relevance_mark = (
            "✓ Relevant"
            if is_relevant
            else "✗ Not Relevant"
        )


        print(
            f"{index}. "
            f"{food_name} "
            f"| {item['recommended_element_th']} "
            f"| score={item['ir_score']} "
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


    print()


# =========================================================
# Main Evaluation
# =========================================================

def run_ir_evaluation():

    print(
        "\n"
        "เริ่มประเมิน Hybrid IR Recommendation System"
        "\n"
    )


    # -----------------------------------------------------
    # Load Data
    # -----------------------------------------------------

    foods = load_csv(
        FOODS_PATH
    )


    test_cases = load_ground_truth()


    print(
        f"โหลด foods.csv: "
        f"{len(foods)} rows"
    )


    print(
        f"โหลด Ground Truth: "
        f"{len(test_cases)} queries"
    )

    print()


    # -----------------------------------------------------
    # Score Lists
    # -----------------------------------------------------

    precision_scores = []

    recall_scores = []

    ap_scores = []

    accuracy_scores = []

    coverage_scores = []


    # -----------------------------------------------------
    # Evaluate Each Query
    # -----------------------------------------------------

    for case in test_cases:

        elements = case[
            "elements"
        ]


        # Ground Truth Set
        relevant_foods = (
            build_relevant_food_set(
                foods=foods,
                relevant_elements=elements,
            )
        )


        # IR Retrieval
        results = (
            retrieve_foods_by_element(
                active_elements=elements,
                top_k=TOP_K,
            )
        )


        # -------------------------------------------------
        # Calculate Metrics
        # -------------------------------------------------

        precision = (
            calculate_precision_at_k(
                results=results,
                relevant_foods=relevant_foods,
                k=TOP_K,
            )
        )


        recall = (
            calculate_recall_at_k(
                results=results,
                relevant_foods=relevant_foods,
                k=TOP_K,
            )
        )


        average_precision = (
            calculate_average_precision_at_k(
                results=results,
                relevant_foods=relevant_foods,
                k=TOP_K,
            )
        )


        accuracy = (
            calculate_top_k_accuracy(
                results=results,
                relevant_foods=relevant_foods,
                k=TOP_K,
            )
        )


        element_coverage = (
            calculate_element_coverage_at_k(
                results=results,
                expected_elements=elements,
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


        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        display_results(
            query_id=case[
                "query_id"
            ],
            query=case[
                "query"
            ],
            elements=elements,
            results=results,
            relevant_foods=relevant_foods,
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            accuracy=accuracy,
            element_coverage=element_coverage,
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


    mean_precision = (
        sum(precision_scores)
        /
        total_cases
    )


    mean_recall = (
        sum(recall_scores)
        /
        total_cases
    )


    mean_average_precision = (
        sum(ap_scores)
        /
        total_cases
    )


    mean_accuracy = (
        sum(accuracy_scores)
        /
        total_cases
    )


    mean_coverage = (
        sum(coverage_scores)
        /
        total_cases
    )


    print(
        "=" * 70
    )

    print(
        "Hybrid IR Evaluation Summary"
    )

    print(
        "=" * 70
    )


    print(
        f"Queries evaluated: "
        f"{total_cases}"
    )


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


    print(
        f"Mean Element Coverage@{TOP_K}: "
        f"{mean_coverage:.3f}"
    )


    print(
        "=" * 70
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    run_ir_evaluation()