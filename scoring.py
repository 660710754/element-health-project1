from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any


# =========================================================
# Path
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

WEIGHTS_FILE = DATA_DIR / "question_weights.csv"



# =========================================================
# Elements
# =========================================================

ELEMENTS = (
    "earth",
    "water",
    "wind",
    "fire",
)


ELEMENT_NAMES_TH = {

    "earth": "ธาตุดิน",

    "water": "ธาตุน้ำ",

    "wind": "ธาตุลม",

    "fire": "ธาตุไฟ",

}



# =========================================================
# Birth Element
# ธาตุเจ้าเรือนจากเดือนเกิด
# =========================================================

def get_birth_element(month: int) -> str:
    """
    คำนวณธาตุเจ้าเรือนจากเดือนเกิด

    ใช้สำหรับแสดงผลเท่านั้น
    ไม่เกี่ยวกับการคำนวณคะแนนแบบสอบถาม
    """

    if not 1 <= month <= 12:
        raise ValueError(
            "เดือนเกิดต้องอยู่ระหว่าง 1-12"
        )


    if month in (1, 2, 3):

        return "fire"


    elif month in (4, 5, 6):

        return "wind"


    elif month in (7, 8, 9):

        return "water"


    else:

        return "earth"



# =========================================================
# Load Weight
# =========================================================

def load_question_weights() -> Dict[str, Dict[str, float]]:

    weights = {}


    if not WEIGHTS_FILE.exists():

        raise FileNotFoundError(
            f"ไม่พบไฟล์ {WEIGHTS_FILE}"
        )


    with WEIGHTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:


        reader = csv.DictReader(file)


        required_columns = {

            "question_id",

            "earth_weight",

            "water_weight",

            "wind_weight",

            "fire_weight",

        }


        missing = (
            required_columns
            -
            set(reader.fieldnames or [])
        )


        if missing:

            raise ValueError(
                f"question_weights.csv ขาด {sorted(missing)}"
            )



        for row in reader:


            question_id = row["question_id"].strip()


            weights[question_id] = {

                "earth":
                    float(row["earth_weight"]),


                "water":
                    float(row["water_weight"]),


                "wind":
                    float(row["wind_weight"]),


                "fire":
                    float(row["fire_weight"]),


            }


    return weights



# =========================================================
# Calculate Current Element Score
# =========================================================

def calculate_scores(
    answers: Dict[str, int]
) -> Dict[str, float]:
    """
    คำนวณธาตุปัจจุบันจากคำตอบแบบสอบถาม

    ไม่ใช้เดือนเกิด
    """

    weights = load_question_weights()


    scores = {

        element: 0.0

        for element in ELEMENTS

    }



    for question_id, answer_score in answers.items():


        if question_id not in weights:

            raise KeyError(
                f"ไม่พบ {question_id}"
            )


        if answer_score not in (0,1,2):

            raise ValueError(
                "คะแนนต้องเป็น 0,1,2"
            )


        for element in ELEMENTS:


            scores[element] += (

                answer_score

                *

                weights[question_id][element]

            )



    return {

        element:

            round(score,2)

        for element,score in scores.items()

    }



# =========================================================
# Confidence
# =========================================================

def calculate_confidence(
    scores: Dict[str,float]
)->float:


    total = sum(scores.values())


    if total == 0:

        return 0.0



    highest = max(scores.values())


    return round(
        highest / total,
        2
    )



# =========================================================
# Confidence Interpretation
# =========================================================

def interpret_confidence(
    confidence: float
)->str:


    if confidence >= 0.75:

        return "มีแนวโน้มธาตุเด่นอย่างชัดเจน"



    elif confidence >= 0.55:

        return (
            "มีแนวโน้มธาตุเด่น "
            "แต่มีลักษณะร่วมกับธาตุอื่น"
        )



    else:

        return (
            "ผลประเมินยังไม่ชัดเจน "
            "ควรเก็บข้อมูลเพิ่มเติม"
        )



# =========================================================
# Result
# =========================================================

def get_result(
    scores: Dict[str,float],
    birth_month: int | None = None
)->Dict[str,Any]:
    """
    สรุปผล

    แสดง:
    - ธาตุเด่นปัจจุบัน
    - ธาตุรอง
    - ธาตุเจ้าเรือนจากเดือนเกิด

    """

    ranking = sorted(

        scores.items(),

        key=lambda item:item[1],

        reverse=True

    )


    primary_element, primary_score = ranking[0]


    secondary_element, secondary_score = ranking[1]



    result = {


        "primary_element":

            primary_element,


        "primary_score":

            primary_score,


        "secondary_element":

            secondary_element,


        "secondary_score":

            secondary_score,


        "confidence":

            calculate_confidence(scores),


        "confidence_level":

            interpret_confidence(
                calculate_confidence(scores)
            ),


        "ranking":

            ranking,

    }



    # เพิ่มธาตุเกิดเฉพาะกรณีมีเดือนเกิด

    if birth_month is not None:

        result["birth_element"] = get_birth_element(
            birth_month
        )

        result["birth_element_th"] = element_name(
            result["birth_element"]
        )

    else:

        result["birth_element"] = None

        result["birth_element_th"] = None



    return result



# =========================================================
# Element Name
# =========================================================

def element_name(
    element:str
)->str:


    return ELEMENT_NAMES_TH.get(
        element,
        element
    )



# =========================================================
# Test
# =========================================================

if __name__ == "__main__":


    sample_answers = {

        "Q002":0,

        "Q003":1,

        "Q004":2,

        "Q005":2,

        "Q006":1,

        "Q007":0,

        "Q008":2,

    }



    scores = calculate_scores(
        sample_answers
    )


    result = get_result(
        scores,
        birth_month=5
    )



    print("="*60)

    print("ผลประเมินธาตุ")

    print("="*60)



    for element,score in result["ranking"]:

        print(
            element_name(element),
            ":",
            score
        )


    print()

    print(
        "ธาตุเด่นปัจจุบัน:",
        element_name(
            result["primary_element"]
        )
    )


    print(
        "ธาตุรอง:",
        element_name(
            result["secondary_element"]
        )
    )


    print(
        "ธาตุเจ้าเรือนเกิด:",
        result["birth_element_th"]
    )


    print(
        "Confidence:",
        result["confidence"]
    )