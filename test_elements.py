import json
from pathlib import Path
from typing import Any


ELEMENTS_FILE = Path("data/elements.json")
REQUIRED_ELEMENTS = {"earth", "water", "wind", "fire"}


def load_json(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError("ข้อมูลระดับบนสุดของ JSON ต้องเป็น object")

    return data


def validate_elements(data: dict[str, Any]) -> None:
    if "metadata" not in data:
        raise ValueError("ไม่พบ metadata")

    if "elements" not in data:
        raise ValueError("ไม่พบ elements")

    elements = data["elements"]

    if not isinstance(elements, dict):
        raise TypeError("elements ต้องเป็น object")

    missing_elements = REQUIRED_ELEMENTS - set(elements.keys())

    if missing_elements:
        raise ValueError(
            f"ข้อมูลธาตุไม่ครบ ขาด: {', '.join(sorted(missing_elements))}"
        )

    required_fields = {
        "element_id",
        "name_th",
        "name_en",
        "thai_medicine_role",
        "general_traits",
        "current_signs",
        "recommended_tastes",
        "recommended_foods",
        "foods_to_avoid",
        "health_considerations",
        "sources"
    }

    for element_id, element in elements.items():
        if not isinstance(element, dict):
            raise TypeError(f"{element_id} ต้องเป็น object")

        missing_fields = required_fields - set(element.keys())

        if missing_fields:
            raise ValueError(
                f"{element_id} ขาดข้อมูล: "
                f"{', '.join(sorted(missing_fields))}"
            )

        if element["element_id"] != element_id:
            raise ValueError(
                f"element_id ของ {element_id} ไม่ตรงกับชื่อ key"
            )

        if not element["current_signs"]:
            raise ValueError(f"{element_id} ไม่มี current_signs")

        if not element["recommended_tastes"]:
            raise ValueError(f"{element_id} ไม่มี recommended_tastes")


def show_summary(data: dict[str, Any]) -> None:
    elements = data["elements"]

    print("โหลดและตรวจสอบข้อมูลสำเร็จ")
    print(f"เวอร์ชันข้อมูล: {data['metadata']['version']}")
    print(f"จำนวนธาตุ: {len(elements)}")
    print()

    for element_id, element in elements.items():
        foods = element["recommended_foods"]

        number_of_food_items = sum(
            len(items)
            for items in foods.values()
            if isinstance(items, list)
        )

        print(f"{element_id}: {element['name_th']}")
        print(f"  ลักษณะทั่วไป: {len(element['general_traits'])} รายการ")
        print(f"  อาการปัจจุบัน: {len(element['current_signs'])} รายการ")
        print(f"  อาหารและเมนู: {number_of_food_items} รายการ")
        print(f"  สิ่งที่ควรหลีกเลี่ยง: {len(element['foods_to_avoid'])} รายการ")


def main() -> None:
    try:
        data = load_json(ELEMENTS_FILE)
        validate_elements(data)
        show_summary(data)

    except json.JSONDecodeError as error:
        print("รูปแบบ JSON ไม่ถูกต้อง")
        print(f"บรรทัดที่ {error.lineno}, คอลัมน์ที่ {error.colno}")
        print(error.msg)

    except (FileNotFoundError, TypeError, ValueError) as error:
        print("ตรวจสอบข้อมูลไม่ผ่าน")
        print(error)

    except Exception as error:
        print("เกิดข้อผิดพลาดที่ไม่คาดคิด")
        print(error)


if __name__ == "__main__":
    main()