import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "questions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "questions_fixed.csv"

OLD_COLUMNS = [
    "question_id",
    "section",
    "question_th",
    "response_type",
    "option_0",
    "option_1",
    "option_2",
    "earth_weight",
    "water_weight",
    "wind_weight",
    "fire_weight",
    "timeframe",
    "is_scored",
    "source_basis",
]

NEW_COLUMNS = [
    "question_id",
    "section",
    "question_th",
    "response_type",
    "option_0",
    "option_1",
    "option_2",
    "timeframe",
    "is_scored",
    "source_basis",
]


def main() -> None:
    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as source:
        # ข้าม Header ที่แก้ไปแล้ว
        reader = csv.reader(source)
        next(reader)

        rows = list(reader)

    fixed_rows = []

    for line_number, row in enumerate(rows, start=2):
        if len(row) != len(OLD_COLUMNS):
            raise ValueError(
                f"บรรทัด {line_number} มี {len(row)} คอลัมน์ "
                f"แต่ควรมี {len(OLD_COLUMNS)} คอลัมน์"
            )

        original = dict(zip(OLD_COLUMNS, row))

        fixed_rows.append({
            column: original[column]
            for column in NEW_COLUMNS
        })

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=NEW_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(fixed_rows)

    print(f"สร้างไฟล์สำเร็จ: {OUTPUT_FILE}")
    print(f"จำนวนข้อมูล: {len(fixed_rows)} ข้อ")


if __name__ == "__main__":
    main()