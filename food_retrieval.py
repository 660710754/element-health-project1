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

ELEMENT_NAME_TH = {
    "earth": "ดิน",
    "water": "น้ำ",
    "wind": "ลม",
    "fire": "ไฟ",
}

DEFAULT_TOP_K = 10

MIN_SIMILARITY = 0.01


# =========================================================
# Category Intent
# =========================================================

CATEGORY_KEYWORDS = {

    "vegetable_herb": (
        "ผัก",
        "สมุนไพร",
        "ผักสมุนไพร",
        "ผักพื้นบ้าน",
    ),

    "fruit": (
        "ผลไม้",
        "ผลไม้สด",
    ),

    "menu": (
        "เมนู",
        "อาหารจาน",
        "อาหารปรุง",
        "กับข้าว",
    ),

    "snack": (
        "ขนม",
        "ของหวาน",
        "ของว่าง",
        "อาหารว่าง",
    ),

    "drink": (
        "เครื่องดื่ม",
        "น้ำสมุนไพร",
        "น้ำผลไม้",
    ),
}


# =========================================================
# Taste Intent
# =========================================================

TASTE_KEYWORDS = {

    "ขม": (
        "รสขม",
        "ขม",
    ),

    "เปรี้ยว": (
        "รสเปรี้ยว",
        "เปรี้ยว",
    ),

    "หวาน": (
        "รสหวาน",
        "หวาน",
    ),

    "เค็ม": (
        "รสเค็ม",
        "เค็ม",
    ),

    "ฝาด": (
        "รสฝาด",
        "ฝาด",
    ),

    "เผ็ด": (
        "รสเผ็ด",
        "เผ็ด",
    ),

    "จืด": (
        "รสจืด",
        "จืด",
    ),

    "มัน": (
        "รสมัน",
    ),

    "หอม": (
        "รสหอม",
        "กลิ่นหอม",
    ),
}


# =========================================================
# Load foods.csv
# =========================================================

def load_food_database() -> list[dict[str, str]]:
    """
    โหลดฐานข้อมูลอาหารจาก data/foods.csv
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

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )

        fieldnames = (
            reader.fieldnames
            or []
        )


    if not rows:

        raise ValueError(
            "foods.csv ไม่มีข้อมูล"
        )


    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = {

        "food_id",

        "food_name_th",

        "category",

        "category_th",

        "recommended_element",

        "recommended_element_th",

        "taste_profile",

        "food_taste_profile",

        "recommendation_status",

        "reason_th",

    }


    missing_columns = (
        required_columns
        -
        set(fieldnames)
    )


    if missing_columns:

        raise ValueError(

            "foods.csv ขาดคอลัมน์: "

            + ", ".join(
                sorted(
                    missing_columns
                )
            )

        )


    cleaned_rows = []


    for row in rows:

        cleaned = {

            key: (
                value or ""
            ).strip()

            for key, value
            in row.items()

        }


        if not cleaned.get(
            "food_id"
        ):
            continue


        if cleaned.get(
            "recommendation_status"
        ) not in {
            "recommended",
            "avoid"
        }:
            continue


        cleaned_rows.append(
            cleaned
        )


    return cleaned_rows


# =========================================================
# Convert food to TF-IDF document
# =========================================================

def create_food_documents(
    foods: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """
    สร้างข้อความสำหรับ TF-IDF

    ใช้ข้อมูล:
    - ชื่ออาหาร
    - หมวดอาหาร
    - ธาตุ
    - รสตามหลักธาตุ
    - รสจริงของอาหาร
    - เหตุผลคำแนะนำ
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
                    "food_taste_profile",
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
                "data": food,
            }
        )


    return documents


# =========================================================
# Detect Category Intent
# =========================================================

def detect_category_intent(
    query: str
) -> str | None:
    """
    ตรวจสอบว่าผู้ใช้ถามถึงหมวดอาหารใด

    ตัวอย่าง:
    "ธาตุไฟควรกินผักอะไร"
        -> vegetable_herb

    "ธาตุน้ำมีผลไม้อะไร"
        -> fruit
    """

    query = (
        query or ""
    ).strip()


    for category, keywords in (
        CATEGORY_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in query:

                return category


    return None


# =========================================================
# Detect Taste Intent
# =========================================================

def detect_taste_intents(
    query: str
) -> list[str]:
    """
    ตรวจหารสชาติที่ผู้ใช้ระบุ

    ตัวอย่าง:
    "ผักรสขม"
        -> ["ขม"]

    "อาหารเปรี้ยวหวาน"
        -> ["เปรี้ยว", "หวาน"]
    """

    query = (
        query or ""
    ).strip()


    found_tastes = []


    for taste, keywords in (
        TASTE_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in query:

                found_tastes.append(
                    taste
                )

                break


    return found_tastes


# =========================================================
# Check taste match
# =========================================================

def food_matches_taste(
    food: dict[str, str],
    tastes: list[str]
) -> bool:
    """
    ตรวจว่า food_taste_profile
    ตรงกับรสชาติที่ผู้ใช้ต้องการหรือไม่

    รองรับค่าประมาณ:
    ขม
    ขมอ่อนๆ
    ขมอมหวาน

    หากผู้ใช้ถาม "ขม"
    ทั้งสามแบบถือว่าตรง
    """

    if not tastes:
        return True


    profile = (
        food.get(
            "food_taste_profile",
            ""
        )
        .strip()
        .lower()
    )


    if not profile:

        return False


    for taste in tastes:

        if (
            taste.lower()
            in profile
        ):

            return True


    return False


# =========================================================
# Food Retriever
# =========================================================

class FoodRetriever:
    """
    Hybrid Information Retrieval

    1. Element Filtering
    2. Category Intent Filtering
    3. Taste Intent Filtering
    4. TF-IDF
    5. Cosine Similarity
    6. Ranking
    """


    def __init__(self):

        self.all_foods = (
            load_food_database()
        )


    # -----------------------------------------------------
    # Validate Elements
    # -----------------------------------------------------

    def validate_elements(
        self,
        elements: list[str]
    ) -> list[str]:

        valid_elements = []


        for element in elements:

            normalized = (
                element
                .strip()
                .lower()
            )


            if (
                normalized in ELEMENTS
                and
                normalized not in valid_elements
            ):

                valid_elements.append(
                    normalized
                )


        if not valid_elements:

            raise ValueError(

                "ไม่พบธาตุที่ถูกต้อง "
                f"รองรับเฉพาะ: {ELEMENTS}"

            )


        return valid_elements


    # -----------------------------------------------------
    # Element Filter
    # -----------------------------------------------------

    def filter_by_element(
        self,
        elements: list[str]
    ) -> list[dict[str, str]]:
        """
        เลือกเฉพาะอาหารที่:
        - recommended_element ตรงกับธาตุ
        - recommendation_status == recommended
        """

        valid_elements = (
            self.validate_elements(
                elements
            )
        )


        foods = [

            food

            for food
            in self.all_foods

            if (

                food.get(
                    "recommended_element",
                    ""
                )
                in valid_elements

                and

                food.get(
                    "recommendation_status",
                    ""
                )
                ==
                "recommended"

            )

        ]


        return foods


    # -----------------------------------------------------
    # Category Filter
    # -----------------------------------------------------

    def filter_by_category(
        self,
        foods: list[dict[str, str]],
        category: str | None
    ) -> list[dict[str, str]]:
        """
        กรองตามหมวดอาหารจาก User Intent
        """

        if not category:
            return foods


        filtered = [

            food

            for food in foods

            if (
                food.get(
                    "category",
                    ""
                ).strip()
                ==
                category
            )

        ]


        # ถ้ากรองแล้วไม่มีข้อมูลเลย
        # ให้กลับไปใช้ candidate เดิม
        # เพื่อไม่ให้ระบบตอบว่างโดยไม่จำเป็น

        if not filtered:
            return foods


        return filtered


    # -----------------------------------------------------
    # Taste Filter
    # -----------------------------------------------------

    def filter_by_taste(
        self,
        foods: list[dict[str, str]],
        tastes: list[str]
    ) -> list[dict[str, str]]:
        """
        กรองตาม food_taste_profile

        ใช้เฉพาะเมื่อ User ระบุรสชาติ
        """

        if not tastes:
            return foods


        filtered = [

            food

            for food in foods

            if food_matches_taste(
                food,
                tastes
            )

        ]


        # หากไม่มีอาหารที่ตรงรสเลย
        # จะไม่ทำให้ Candidate หายทั้งหมด
        # แต่กลับไป Ranking จาก Candidate ก่อนหน้า

        if not filtered:
            return foods


        return filtered


    # -----------------------------------------------------
    # Apply Query Intent
    # -----------------------------------------------------

    def apply_query_intent(
        self,
        foods: list[dict[str, str]],
        query: str
    ) -> tuple[
        list[dict[str, str]],
        str | None,
        list[str]
    ]:
        """
        ตรวจ Category + Taste
        จาก User Input
        """

        category = (
            detect_category_intent(
                query
            )
        )


        tastes = (
            detect_taste_intents(
                query
            )
        )


        candidates = (
            self.filter_by_category(
                foods=foods,
                category=category
            )
        )


        candidates = (
            self.filter_by_taste(
                foods=candidates,
                tastes=tastes
            )
        )


        return (
            candidates,
            category,
            tastes
        )


    # -----------------------------------------------------
    # TF-IDF Ranking
    # -----------------------------------------------------

    def rank_by_similarity(
        self,
        foods: list[dict[str, str]],
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:
        """
        จัดอันดับ Candidate Foods
        ด้วย TF-IDF + Cosine Similarity
        """

        if not foods:
            return []


        query = (
            query or ""
        ).strip()


        if not query:

            raise ValueError(
                "query ต้องไม่เป็นค่าว่าง"
            )


        if top_k <= 0:

            raise ValueError(
                "top_k ต้องมากกว่า 0"
            )


        documents = (
            create_food_documents(
                foods
            )
        )


        texts = [

            item["text"]

            for item
            in documents

        ]


        vectorizer = (
            TfidfVectorizer(

                analyzer="char",

                ngram_range=(2, 6),

                lowercase=False,

                sublinear_tf=True,

                norm="l2",

            )
        )


        matrix = (
            vectorizer.fit_transform(
                texts
            )
        )


        query_vector = (
            vectorizer.transform(
                [query]
            )
        )


        scores = (
            cosine_similarity(
                query_vector,
                matrix
            )[0]
        )


        ranked_indexes = (
            scores.argsort()[::-1]
        )


        results = []


        for index in ranked_indexes:

            score = float(
                scores[index]
            )


            food = dict(
                documents[
                    index
                ][
                    "data"
                ]
            )


            food[
                "ir_score"
            ] = round(
                score,
                4
            )


            results.append(
                food
            )


            if (
                len(results)
                >=
                top_k
            ):
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

        ขั้นตอน:

        Element
        ↓
        Category Intent
        ↓
        Taste Intent
        ↓
        TF-IDF
        ↓
        Cosine Similarity
        ↓
        Top-K
        """

        # ---------------------------------------------
        # 1. Element Filtering
        # ---------------------------------------------

        candidates = (
            self.filter_by_element(
                elements
            )
        )


        # ---------------------------------------------
        # 2. User Query Intent
        # ---------------------------------------------

        (
            candidates,
            detected_category,
            detected_tastes

        ) = self.apply_query_intent(

            foods=candidates,

            query=query

        )


        # ---------------------------------------------
        # 3. TF-IDF Ranking
        # ---------------------------------------------

        results = (
            self.rank_by_similarity(

                foods=candidates,

                query=query,

                top_k=top_k

            )
        )


        # ---------------------------------------------
        # เก็บข้อมูล Intent
        # สำหรับ Debug / Evaluation
        # ---------------------------------------------

        for result in results:

            result[
                "detected_category"
            ] = (
                detected_category
                or ""
            )


            result[
                "detected_tastes"
            ] = "|".join(
                detected_tastes
            )


        return results


# =========================================================
# User Input Retrieval
# =========================================================

def retrieve_foods(
    query: str,
    active_elements: list[str],
    top_k: int = DEFAULT_TOP_K
) -> list[dict[str, Any]]:
    """
    รับ User Input จริง

    ใช้สำหรับ:
    - Ground Truth Evaluation
    - IR Testing
    - UI ในอนาคต

    ตัวอย่าง:

    retrieve_foods(
        query="ธาตุไฟควรกินผักรสขมอะไร",
        active_elements=["fire"],
        top_k=5
    )
    """

    retriever = (
        FoodRetriever()
    )


    return retriever.search(

        elements=active_elements,

        query=query,

        top_k=top_k

    )


# =========================================================
# Recommendation Compatibility
# =========================================================

def retrieve_foods_by_element(
    active_elements: list[str],
    top_k: int = DEFAULT_TOP_K
) -> list[dict[str, Any]]:
    """
    สำหรับ recommendation.py เดิม

    ยังคง Function เดิมไว้
    เพื่อไม่ให้ระบบ Recommendation พัง
    """

    retriever = (
        FoodRetriever()
    )


    query_terms = []


    for element in active_elements:

        normalized = (
            element
            .strip()
            .lower()
        )


        element_th = (
            ELEMENT_NAME_TH.get(
                normalized
            )
        )


        if element_th:

            query_terms.append(
                f"อาหารธาตุ{element_th}"
            )


    if not query_terms:

        raise ValueError(
            "active_elements "
            "ไม่มีธาตุที่ถูกต้อง"
        )


    query = " ".join(
        query_terms
    )


    return retriever.search(

        elements=active_elements,

        query=query,

        top_k=top_k

    )


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    tests = [

        (
            ["fire"],
            "อาหารธาตุไฟ"
        ),

        (
            ["water"],
            "อาหารธาตุน้ำ"
        ),

        (
            ["earth"],
            "อาหารธาตุดิน"
        ),

        (
            ["wind"],
            "อาหารธาตุลม"
        ),

        (
            ["fire"],
            "ธาตุไฟควรกินผักรสขมอะไร"
        ),

        (
            ["water"],
            "ธาตุน้ำมีผลไม้รสเปรี้ยวอะไร"
        ),

        (
            ["wind"],
            "ธาตุลมมีเมนูรสเผ็ดอะไร"
        ),

        (
            ["earth"],
            "ธาตุดินมีขนมรสหวานอะไร"
        ),

    ]


    retriever = (
        FoodRetriever()
    )


    for elements, query in tests:

        print(
            "=" * 75
        )

        print(
            "ELEMENT:",
            elements
        )

        print(
            "QUERY:",
            query
        )

        print(
            "CATEGORY INTENT:",
            detect_category_intent(
                query
            )
        )

        print(
            "TASTE INTENT:",
            detect_taste_intents(
                query
            )
        )

        print(
            "=" * 75
        )


        results = (
            retriever.search(

                elements=elements,

                query=query,

                top_k=5

            )
        )


        if not results:

            print(
                "ไม่พบผลลัพธ์"
            )

            print()

            continue


        for index, food in enumerate(
            results,
            start=1
        ):

            print(

                index,

                food[
                    "food_name_th"
                ],

                "|",

                food[
                    "recommended_element_th"
                ],

                "| category:",

                food.get(
                    "category",
                    ""
                ),

                "| taste:",

                food.get(
                    "food_taste_profile",
                    ""
                ),

                "| score:",

                food[
                    "ir_score"
                ],

            )


        print()