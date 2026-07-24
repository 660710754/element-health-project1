from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

QUESTIONS_FILE = DATA_DIR / "questions.csv"
WEIGHTS_FILE = DATA_DIR / "question_weights.csv"

ELEMENTS = ("earth", "water", "wind", "fire")


def load_question_weights() -> Dict[str, Dict[str, float]]:
    """โหลดค่าน้ำหนักของแต่ละคำถามจาก question_weights.csv"""
    weights: Dict[str, Dict[str, float]] = {}

    with WEIGHTS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "question_id",
            "earth_weight",
            "water_weight",
            "wind_weight",
            "fire_weight",
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"question_weights.csv ขาดคอลัมน์: {sorted(missing)}"
            )

        for row in reader:
            question_id = row["question_id"].strip()

            weights[question_id] = {
                "earth": float(row["earth_weight"]),
                "water": float(row["water_weight"]),
                "wind": float(row["wind_weight"]),
                "fire": float(row["fire_weight"]),
            }

    return weights


def get_birth_element(month: int) -> str:
    """
    คำนวณธาตุเกิดจากเดือนเกิด

    ม.ค.–มี.ค.   = ไฟ
    เม.ย.–มิ.ย. = ลม
    ก.ค.–ก.ย.   = น้ำ
    ต.ค.–ธ.ค.   = ดิน
    """
    if not 1 <= month <= 12:
        raise ValueError("เดือนเกิดต้องอยู่ระหว่าง 1 ถึง 12")

    if month in (1, 2, 3):
        return "fire"
    if month in (4, 5, 6):
        return "wind"
    if month in (7, 8, 9):
        return "water"

    return "earth"


def calculate_scores(
    answers: Dict[str, int],
    birth_month: int,
    birth_bonus: float = 2.0,
) -> Dict[str, float]:
    """
    คำนวณคะแนนธาตุ

    answers ตัวอย่าง:
    {
        "Q002": 2,
        "Q003": 1,
        "Q004": 0
    }

    คะแนนคำตอบ:
    0 = ไม่ตรง
    1 = ปานกลาง
    2 = ตรงมาก
    """
    weights = load_question_weights()

    scores = {element: 0.0 for element in ELEMENTS}

    for question_id, answer_score in answers.items():
        if question_id not in weights:
            raise KeyError(f"ไม่พบ {question_id} ใน question_weights.csv")

        if answer_score not in (0, 1, 2):
            raise ValueError(
                f"คำตอบของ {question_id} ต้องเป็น 0, 1 หรือ 2"
            )

        for element in ELEMENTS:
            scores[element] += (
                answer_score * weights[question_id][element]
            )

    birth_element = get_birth_element(birth_month)
    scores[birth_element] += birth_bonus

    return {
        element: round(score, 2)
        for element, score in scores.items()
    }


def get_result(scores: Dict[str, float]) -> Dict[str, object]:
    """เรียงคะแนนและหาธาตุเด่นกับธาตุรอง"""
    ranking = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    primary_element, primary_score = ranking[0]
    secondary_element, secondary_score = ranking[1]

    return {
        "primary_element": primary_element,
        "primary_score": primary_score,
        "secondary_element": secondary_element,
        "secondary_score": secondary_score,
        "ranking": ranking,
    }


if __name__ == "__main__":
    # ข้อมูลจำลองสำหรับทดสอบระบบ
    sample_answers = {
        "Q002": 0,
        "Q003": 0,
        "Q004": 0,
        "Q005": 1,
        "Q006": 0,
        "Q007": 0,
        "Q008": 0,
        "Q009": 0,
        "Q010": 0,
        "Q011": 0,
        "Q012": 0,
        "Q013": 0,

        "Q014": 0,
        "Q015": 0,
        "Q016": 0,
        "Q017": 0,
        "Q018": 0,
        "Q019": 0,
        "Q020": 0,
        "Q021": 0,
        "Q022": 0,
        "Q023": 0,
        "Q024": 0,
        "Q025": 0,

        "Q026": 2,
        "Q027": 2,
        "Q028": 2,
        "Q029": 2,
        "Q030": 2,
        "Q031": 1,
        "Q032": 2,
        "Q033": 2,
        "Q034": 2,
        "Q035": 2,
        "Q036": 2,
        "Q037": 1,

        "Q038": 0,
        "Q039": 0,
        "Q040": 0,
        "Q041": 1,
        "Q042": 0,
        "Q043": 0,
        "Q044": 0,
        "Q045": 0,
        "Q046": 0,
        "Q047": 0,
        "Q048": 0,
        "Q049": 0,
    }

    scores = calculate_scores(
        answers=sample_answers,
        birth_month=5,
    )

    result = get_result(scores)

    print("คะแนนรวม")
    for element, score in result["ranking"]:
        print(f"{element}: {score}")

    print()
    print("ธาตุเด่น:", result["primary_element"])
    print("ธาตุรอง:", result["secondary_element"])