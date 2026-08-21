from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from recommendation import (
    DEFAULT_MIXED_THRESHOLD,
    recommend_foods,
)

from rag import (
    DEFAULT_MODEL,
    RAGSystem,
)


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

COMPARISON_CASES_PATH = (
    BASE_DIR
    / "data"
    / "ir_rag_comparison_cases.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "ir_rag_comparison_results.csv"
)

DEFAULT_TOP_K = 5


# =========================================================
# Required Columns
# =========================================================

REQUIRED_COLUMNS = {
    "case_id",
    "query",
    "comparison_type",
    "primary_element",
    "secondary_element",
    "active_elements",
    "category",
    "top_k",
}


# =========================================================
# Category Mapping
# =========================================================

IR_CATEGORY_MAPPING = {

    # คำถาม "อาหารอะไร"
    # ใช้ผัก/สมุนไพร + ผลไม้
    "food": [
        "vegetable_herb",
        "fruit",
    ],

    # คำถามเกี่ยวกับเมนู
    "menu": [
        "menu",
    ],
}


# =========================================================
# Element Configuration
# =========================================================

ELEMENTS = [
    "earth",
    "water",
    "wind",
    "fire",
]


# =========================================================
# Load CSV
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


    if reader.fieldnames is None:

        raise ValueError(
            f"ไฟล์ {path.name} ไม่มี Header"
        )


    missing_columns = (
        REQUIRED_COLUMNS
        -
        set(
            reader.fieldnames
        )
    )


    if missing_columns:

        raise ValueError(
            "Comparison CSV ขาดคอลัมน์: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )


    return rows


# =========================================================
# Parse Pipe Values
# =========================================================

def parse_pipe_values(
    value: str,
) -> list[str]:
    """
    water|earth

    ->

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
# Load Comparison Cases
# =========================================================

def load_comparison_cases() -> list[
    dict[str, Any]
]:
    """
    โหลด Comparison Cases
    """

    rows = load_csv(
        COMPARISON_CASES_PATH
    )


    cases: list[
        dict[str, Any]
    ] = []


    for row in rows:

        top_k_text = (
            row.get(
                "top_k",
                "",
            )
            .strip()
        )


        top_k = (
            int(
                top_k_text
            )
            if top_k_text
            else DEFAULT_TOP_K
        )


        if top_k <= 0:

            raise ValueError(
                f"{row['case_id']}: "
                "top_k ต้องมากกว่า 0"
            )


        active_elements = (
            parse_pipe_values(
                row[
                    "active_elements"
                ]
            )
        )


        for element in active_elements:

            if element not in ELEMENTS:

                raise ValueError(
                    f"{row['case_id']}: "
                    f"ไม่รู้จักธาตุ {element}"
                )


        cases.append(
            {
                "case_id": (
                    row[
                        "case_id"
                    ].strip()
                ),

                "query": (
                    row[
                        "query"
                    ].strip()
                ),

                "comparison_type": (
                    row[
                        "comparison_type"
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

                "active_elements": (
                    active_elements
                ),

                "category": (
                    row[
                        "category"
                    ].strip()
                ),

                "top_k": (
                    top_k
                ),
            }
        )


    return cases


# =========================================================
# Build IR Scores
# =========================================================

def build_ir_scores(
    case: dict[str, Any],
) -> dict[str, float]:
    """
    สร้าง scores สำหรับส่งให้ recommendation.py


    SINGLE
    -------

    ตัวอย่าง fire:

    fire  = 10
    earth = 0
    water = 0
    wind  = 0

    -> primary_only


    MIXED
    -----

    ตัวอย่าง:

    primary   = water
    secondary = earth

    water = 10.0
    earth = 9.5

    difference = 0.5

    เพราะ DEFAULT_MIXED_THRESHOLD = 1.0

    -> mixed
    """

    scores = {
        element: 0.0
        for element in ELEMENTS
    }


    comparison_type = (
        case[
            "comparison_type"
        ]
    )


    active_elements = (
        case[
            "active_elements"
        ]
    )


    primary_element = (
        case[
            "primary_element"
        ]
    )


    secondary_element = (
        case[
            "secondary_element"
        ]
    )


    # -----------------------------------------------------
    # Single Element
    # -----------------------------------------------------

    if comparison_type == "single":

        if not active_elements:

            raise ValueError(
                f"{case['case_id']}: "
                "single case ไม่มี active element"
            )


        target_element = (
            active_elements[0]
        )


        scores[
            target_element
        ] = 10.0


        return scores


    # -----------------------------------------------------
    # Mixed Element
    # -----------------------------------------------------

    if comparison_type == "mixed":

        if not primary_element:

            raise ValueError(
                f"{case['case_id']}: "
                "mixed case ไม่มี primary_element"
            )


        if not secondary_element:

            raise ValueError(
                f"{case['case_id']}: "
                "mixed case ไม่มี secondary_element"
            )


        scores[
            primary_element
        ] = 10.0


        # ต่างกัน 0.5
        # จึงอยู่ใน mixed threshold = 1.0
        scores[
            secondary_element
        ] = 9.5


        return scores


    raise ValueError(
        f"{case['case_id']}: "
        f"comparison_type ไม่ถูกต้อง "
        f"{comparison_type!r}"
    )


# =========================================================
# Get IR Categories
# =========================================================

def get_ir_categories(
    comparison_category: str,
) -> list[str]:
    """
    Mapping logical comparison category
    ไปเป็น category จริงใน foods.csv
    """

    if (
        comparison_category
        not in IR_CATEGORY_MAPPING
    ):

        raise ValueError(
            "ไม่รองรับ comparison category: "
            f"{comparison_category}"
        )


    return (
        IR_CATEGORY_MAPPING[
            comparison_category
        ]
    )


# =========================================================
# Run IR
# =========================================================

def run_ir(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    รันระบบ IR / Recommendation
    """

    scores = build_ir_scores(
        case
    )


    categories = (
        get_ir_categories(
            case[
                "category"
            ]
        )
    )


    results = recommend_foods(
        scores=scores,
        limit=case[
            "top_k"
        ],
        categories=categories,
        mixed_threshold=(
            DEFAULT_MIXED_THRESHOLD
        ),
    )


    # -----------------------------------------------------
    # Extract IR Output
    # -----------------------------------------------------

    items: list[str] = []

    elements: list[str] = []

    item_categories: list[str] = []


    for item in results:

        food_name = str(
            item.get(
                "food_name_th",
                "",
            )
        ).strip()


        element = str(
            item.get(
                "recommended_element",
                "",
            )
        ).strip()


        category = str(
            item.get(
                "category",
                "",
            )
        ).strip()


        if food_name:

            items.append(
                food_name
            )


        if element:

            elements.append(
                element
            )


        if category:

            item_categories.append(
                category
            )


    return {
        "items": (
            items
        ),

        "elements": (
            elements
        ),

        "categories": (
            item_categories
        ),

        "raw_results": (
            results
        ),
    }


# =========================================================
# Run RAG
# =========================================================

def run_rag(
    rag: RAGSystem,
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    รัน RAG ด้วย Query เดียวกับ IR
    """

    result = rag.answer(
        question=case[
            "query"
        ],

        primary_element=case[
            "primary_element"
        ],

        secondary_element=case[
            "secondary_element"
        ],
    )


    return result


# =========================================================
# Check RAG Status
# =========================================================

def get_rag_status(
    result: dict[str, Any],
) -> str:
    """
    ตรวจว่า Generation สำเร็จหรือไม่
    """

    answer = str(
        result.get(
            "answer",
            "",
        )
    ).strip()


    if not answer:

        return "empty_answer"


    if (
        "ไม่สามารถเชื่อมต่อกับ Ollama"
        in answer
    ):

        return "ollama_error"


    if (
        "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"
        in answer
    ):

        return "no_information"


    return "success"


# =========================================================
# Join Values
# =========================================================

def join_values(
    values: list[Any],
) -> str:
    """
    รวม list สำหรับเก็บใน CSV

    ใช้ | เป็น separator
    """

    return " | ".join(
        str(value)
        for value in values
        if str(value).strip()
    )


# =========================================================
# Build Output Row
# =========================================================

def build_output_row(
    case: dict[str, Any],
    ir_result: dict[str, Any],
    rag_result: dict[str, Any],
) -> dict[str, Any]:
    """
    รวมผล IR และ RAG
    เป็น 1 row
    """

    rag_sources = (
        rag_result.get(
            "sources",
            [],
        )
    )


    rag_active_elements = (
        rag_result.get(
            "active_elements",
            [],
        )
    )


    rag_query_intents = (
        rag_result.get(
            "query_intents",
            [],
        )
    )


    return {
        "case_id": (
            case[
                "case_id"
            ]
        ),

        "query": (
            case[
                "query"
            ]
        ),

        "comparison_type": (
            case[
                "comparison_type"
            ]
        ),

        "primary_element": (
            case[
                "primary_element"
            ]
            or ""
        ),

        "secondary_element": (
            case[
                "secondary_element"
            ]
            or ""
        ),

        "active_elements": (
            join_values(
                case[
                    "active_elements"
                ]
            )
        ),

        "category": (
            case[
                "category"
            ]
        ),

        "top_k": (
            case[
                "top_k"
            ]
        ),

        # ---------------------------------------------
        # IR
        # ---------------------------------------------

        "ir_items": (
            join_values(
                ir_result[
                    "items"
                ]
            )
        ),

        "ir_elements": (
            join_values(
                ir_result[
                    "elements"
                ]
            )
        ),

        "ir_categories": (
            join_values(
                ir_result[
                    "categories"
                ]
            )
        ),

        # ---------------------------------------------
        # RAG
        # ---------------------------------------------

        "rag_answer": (
            str(
                rag_result.get(
                    "answer",
                    "",
                )
            ).strip()
        ),

        "rag_sources": (
            join_values(
                rag_sources
            )
        ),

        "rag_active_elements": (
            join_values(
                rag_active_elements
            )
        ),

        "rag_query_intents": (
            join_values(
                rag_query_intents
            )
        ),

        "rag_status": (
            get_rag_status(
                rag_result
            )
        ),
    }


# =========================================================
# Save Comparison Results
# =========================================================

def save_results(
    rows: list[
        dict[str, Any]
    ],
) -> None:
    """
    Export comparison results เป็น CSV

    ใช้ utf-8-sig
    เพื่อเปิดภาษาไทยใน Excel ได้ง่าย
    """

    if not rows:

        raise ValueError(
            "ไม่มีผลลัพธ์สำหรับ Export"
        )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fieldnames = [
        "case_id",
        "query",
        "comparison_type",
        "primary_element",
        "secondary_element",
        "active_elements",
        "category",
        "top_k",

        "ir_items",
        "ir_elements",
        "ir_categories",

        "rag_answer",
        "rag_sources",
        "rag_active_elements",
        "rag_query_intents",
        "rag_status",
    ]


    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )


        writer.writeheader()


        writer.writerows(
            rows
        )


# =========================================================
# Display Case
# =========================================================

def display_case(
    case: dict[str, Any],
    ir_result: dict[str, Any],
    rag_result: dict[str, Any],
) -> None:
    """
    แสดงผลแต่ละ Case ใน Terminal
    """

    print(
        "=" * 80
    )


    print(
        f"CASE  : "
        f"{case['case_id']}"
    )


    print(
        f"QUERY : "
        f"{case['query']}"
    )


    print(
        f"TYPE  : "
        f"{case['comparison_type']}"
    )


    print(
        f"CATEGORY : "
        f"{case['category']}"
    )


    print(
        "=" * 80
    )


    # -----------------------------------------------------
    # IR
    # -----------------------------------------------------

    print()

    print(
        "IR OUTPUT"
    )

    print(
        "-" * 80
    )


    if not ir_result[
        "items"
    ]:

        print(
            "ไม่พบผลลัพธ์ IR"
        )

    else:

        for index, (
            food_name,
            element,
        ) in enumerate(
            zip(
                ir_result[
                    "items"
                ],
                ir_result[
                    "elements"
                ],
            ),
            start=1,
        ):

            print(
                f"{index}. "
                f"{food_name} "
                f"| {element}"
            )


    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    print()

    print(
        "RAG OUTPUT"
    )

    print(
        "-" * 80
    )


    print(
        rag_result.get(
            "answer",
            "",
        )
    )


    print()

    print(
        "RAG SOURCES"
    )

    print(
        "-" * 80
    )


    sources = (
        rag_result.get(
            "sources",
            [],
        )
    )


    if not sources:

        print(
            "ไม่พบ Source"
        )

    else:

        for source in sources:

            print(
                f"- {source}"
            )


    print()

    print(
        "RAG STATUS:"
        f" {get_rag_status(rag_result)}"
    )


    print()


# =========================================================
# Generate Comparison Dataset
# =========================================================

def generate_comparison() -> None:
    """
    Main Comparison Generator

    1. อ่าน Comparison Cases
    2. รัน IR
    3. รัน RAG
    4. รวมผล
    5. Export CSV
    """

    print()

    print(
        "=" * 80
    )

    print(
        "Generate IR vs RAG Comparison Dataset"
    )

    print(
        "=" * 80
    )


    cases = (
        load_comparison_cases()
    )


    print(
        f"โหลด Comparison Cases: "
        f"{len(cases)} cases"
    )

    print()


    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    rag = RAGSystem(
        model=DEFAULT_MODEL,
        top_k=DEFAULT_TOP_K,
    )


    output_rows: list[
        dict[str, Any]
    ] = []


    success_count = 0

    rag_error_count = 0


    # -----------------------------------------------------
    # Run Cases
    # -----------------------------------------------------

    for case in cases:

        # ---------------------------------------------
        # IR
        # ---------------------------------------------

        ir_result = run_ir(
            case
        )


        # ---------------------------------------------
        # RAG
        # ---------------------------------------------

        rag_result = run_rag(
            rag=rag,
            case=case,
        )


        # ---------------------------------------------
        # Display
        # ---------------------------------------------

        display_case(
            case=case,
            ir_result=ir_result,
            rag_result=rag_result,
        )


        # ---------------------------------------------
        # Build CSV Row
        # ---------------------------------------------

        output_row = (
            build_output_row(
                case=case,
                ir_result=(
                    ir_result
                ),
                rag_result=(
                    rag_result
                ),
            )
        )


        output_rows.append(
            output_row
        )


        # ---------------------------------------------
        # Status
        # ---------------------------------------------

        rag_status = (
            output_row[
                "rag_status"
            ]
        )


        if rag_status == "success":

            success_count += 1

        else:

            rag_error_count += 1


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_results(
        output_rows
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "IR vs RAG Comparison Summary"
    )

    print(
        "=" * 80
    )


    print(
        f"Cases Processed: "
        f"{len(cases)}"
    )


    print(
        f"RAG Success: "
        f"{success_count}"
    )


    print(
        f"RAG Error / No Information: "
        f"{rag_error_count}"
    )


    print(
        f"Output File: "
        f"{OUTPUT_PATH}"
    )


    print(
        "=" * 80
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    generate_comparison()