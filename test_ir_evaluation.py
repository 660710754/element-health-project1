from __future__ import annotations

from food_retrieval import retrieve_foods_by_element


# =========================================================
# Configuration
# =========================================================

TOP_K = 5


# =========================================================
# Evaluation Dataset
# =========================================================

TEST_CASES = [

    {
        "name": "Primary Fire",
        "elements": [
            "fire"
        ],
        "query": "อาหารธาตุไฟ",
    },

    {
        "name": "Primary Water",
        "elements": [
            "water"
        ],
        "query": "อาหารธาตุน้ำ",
    },

    {
        "name": "Primary Earth",
        "elements": [
            "earth"
        ],
        "query": "อาหารธาตุดิน",
    },

    {
        "name": "Primary Wind",
        "elements": [
            "wind"
        ],
        "query": "อาหารธาตุลม",
    },

    {
        "name": "Mixed Fire Water",
        "elements": [
            "fire",
            "water"
        ],
        "query": "อาหารธาตุไฟและธาตุน้ำ",
    },

    {
        "name": "Equal Earth Water",
        "elements": [
            "earth",
            "water"
        ],
        "query": "อาหารธาตุดินและธาตุน้ำ",
    },

]


# =========================================================
# Metrics
# =========================================================

def calculate_precision_at_k(
    results: list[dict],
    expected_elements: list[str],
    k: int
) -> float:
    """
    Precision@K

    จำนวนผลลัพธ์ที่ถูกต้องใน Top K / K
    """


    if not results:
        return 0.0


    correct = 0


    for item in results[:k]:

        if (
            item["recommended_element"]
            in expected_elements
        ):
            correct += 1


    return correct / min(
        k,
        len(results)
    )



def calculate_top_k_accuracy(
    results: list[dict],
    expected_elements: list[str]
) -> float:
    """
    Top-K Accuracy

    มีผลลัพธ์ถูกต้องอย่างน้อย 1 รายการ = 1
    """

    for item in results:

        if (
            item["recommended_element"]
            in expected_elements
        ):
            return 1.0


    return 0.0



# =========================================================
# Display
# =========================================================

def display_results(
    case_name: str,
    query: str,
    results: list[dict],
    precision: float,
    accuracy: float,
):

    print("=" * 70)

    print(
        f"CASE : {case_name}"
    )

    print(
        f"QUERY : {query}"
    )

    print("=" * 70)


    if not results:

        print(
            "ไม่พบผลลัพธ์"
        )

        return


    for index, item in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"{item['food_name_th']} "
            f"| {item['recommended_element_th']} "
            f"| score={item['ir_score']}"
        )


    print()

    print(
        f"Precision@{TOP_K}: "
        f"{precision:.3f}"
    )


    print(
        f"Top-K Accuracy: "
        f"{accuracy:.3f}"
    )

    print()



# =========================================================
# Main Evaluation
# =========================================================

def run_ir_evaluation():

    precision_scores = []

    accuracy_scores = []


    print(
        "\nเริ่มประเมิน Hybrid IR Recommendation System\n"
    )


    for case in TEST_CASES:


        elements = case["elements"]


        results = retrieve_foods_by_element(
            active_elements=elements,
            top_k=TOP_K
        )


        precision = calculate_precision_at_k(
            results,
            elements,
            TOP_K
        )


        accuracy = calculate_top_k_accuracy(
            results,
            elements
        )


        precision_scores.append(
            precision
        )


        accuracy_scores.append(
            accuracy
        )


        display_results(
            case_name=case["name"],
            query=case["query"],
            results=results,
            precision=precision,
            accuracy=accuracy
        )



    average_precision = (
        sum(precision_scores)
        /
        len(precision_scores)
    )


    average_accuracy = (
        sum(accuracy_scores)
        /
        len(accuracy_scores)
    )


    print("=" * 70)

    print(
        "Hybrid IR Evaluation Summary"
    )

    print("=" * 70)


    print(
        f"Average Precision@{TOP_K}: "
        f"{average_precision:.3f}"
    )


    print(
        "Average Top-K Accuracy: "
        f"{average_accuracy:.3f}"
    )


    print("=" * 70)



# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    run_ir_evaluation()