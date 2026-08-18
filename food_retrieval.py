from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Path
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

FOODS_FILE = BASE_DIR / "data" / "foods.csv"


# =========================================================
# Config
# =========================================================

ELEMENTS = (
    "earth",
    "water",
    "wind",
    "fire",
)

DEFAULT_TOP_K = 10

MIN_SIMILARITY = 0.01



# =========================================================
# Load foods.csv
# =========================================================

def load_food_database() -> list[dict[str, str]]:
    """
    โหลดฐานข้อมูลอาหาร
    """

    if not FOODS_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {FOODS_FILE}"
        )


    with FOODS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))


    if not rows:
        raise ValueError(
            "foods.csv ไม่มีข้อมูล"
        )


    cleaned_rows = []


    for row in rows:

        cleaned = {
            key: (value or "").strip()
            for key, value in row.items()
        }


        if not cleaned.get("food_id"):
            continue


        if cleaned.get(
            "recommendation_status"
        ) not in {
            "recommended",
            "avoid"
        }:
            continue


        cleaned_rows.append(cleaned)


    return cleaned_rows



# =========================================================
# Convert food to document
# =========================================================

def create_food_documents(
    foods: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """
    สร้างข้อความสำหรับ TF-IDF
    """


    documents = []


    for food in foods:


        text = " ".join(
            [
                food.get(
                    "food_name_th",
                    ""
                ),

                food.get(
                    "category_th",
                    ""
                ),

                food.get(
                    "recommended_element_th",
                    ""
                ),

                food.get(
                    "taste_profile",
                    ""
                ),

                food.get(
                    "reason_th",
                    ""
                ),
            ]
        )


        documents.append(
            {
                "text": text,
                "data": food
            }
        )


    return documents



# =========================================================
# Food Retriever
# =========================================================

class FoodRetriever:
    """
    Hybrid IR

    1. Element Filtering
    2. TF-IDF Ranking
    """


    def __init__(self):

        self.all_foods = load_food_database()



    # -----------------------------------------------------
    # Filter Candidate
    # -----------------------------------------------------

    def filter_by_element(
        self,
        elements: list[str]
    ) -> list[dict[str, str]]:
        """
        เลือกเฉพาะอาหารตามธาตุ
        """

        foods = [
            food
            for food in self.all_foods
            if (
                food["recommended_element"]
                in elements
                and
                food["recommendation_status"]
                ==
                "recommended"
            )
        ]


        return foods



    # -----------------------------------------------------
    # TF-IDF Ranking
    # -----------------------------------------------------

    def rank_by_similarity(
        self,
        foods: list[dict[str, str]],
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:

        if not foods:
            return []


        documents = create_food_documents(
            foods
        )


        texts = [
            item["text"]
            for item in documents
        ]


        vectorizer = TfidfVectorizer(

            analyzer="char",

            ngram_range=(2,6),

            lowercase=False,

            sublinear_tf=True,

            norm="l2"

        )


        matrix = vectorizer.fit_transform(
            texts
        )


        query_vector = vectorizer.transform(
            [query]
        )


        scores = cosine_similarity(
            query_vector,
            matrix
        )[0]


        ranked_indexes = (
            scores.argsort()[::-1]
        )


        results = []


        for index in ranked_indexes:


            food = dict(
                documents[index]["data"]
            )


            food["ir_score"] = round(
                float(scores[index]),
                4
            )


            results.append(
                food
            )


            if len(results) >= top_k:
                break



        return results



    # -----------------------------------------------------
    # Main Search
    # -----------------------------------------------------

    def search(
        self,
        elements: list[str],
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:
        """
        Hybrid Retrieval

        elements:
        ["fire"]

        query:
        "อาหาร ธาตุไฟ"
        """


        candidates = self.filter_by_element(
            elements
        )


        results = self.rank_by_similarity(
            foods=candidates,
            query=query,
            top_k=top_k
        )


        return results



# =========================================================
# Function สำหรับเชื่อม recommendation.py
# =========================================================

def retrieve_foods_by_element(
    active_elements: list[str],
    top_k: int = DEFAULT_TOP_K
) -> list[dict[str, Any]]:
    """
    เรียกใช้จากระบบ Recommendation

    ตัวอย่าง:

    ["fire"]

    หรือ

    ["earth","water"]

    """


    retriever = FoodRetriever()


    query = " ".join(
        [
            f"ธาตุ{element}"
            for element in active_elements
        ]
    )


    results = retriever.search(
        elements=active_elements,
        query=query,
        top_k=top_k
    )


    return results



# =========================================================
# Test
# =========================================================

if __name__ == "__main__":


    retriever = FoodRetriever()


    tests = [

        (
            ["fire"],
            "อาหาร ธาตุไฟ"
        ),

        (
            ["water"],
            "อาหาร ธาตุน้ำ"
        ),

        (
            ["earth"],
            "อาหาร ธาตุดิน"
        ),

        (
            ["wind"],
            "อาหาร ธาตุลม"
        ),

    ]



    for elements, query in tests:


        print("=" * 60)

        print(
            "ELEMENT:",
            elements
        )

        print("=" * 60)


        results = retriever.search(
            elements=elements,
            query=query,
            top_k=5
        )


        for index, food in enumerate(
            results,
            start=1
        ):

            print(
                index,
                food["food_name_th"],
                "|",
                food["recommended_element_th"],
                "| score:",
                food["ir_score"]
            )


        print()