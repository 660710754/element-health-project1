from __future__ import annotations

from pathlib import Path
from typing import Any

import ollama

from knowledge_retrieval import (
    ELEMENT_NAMES_TH,
    build_knowledge_context,
    detect_explicit_elements,
    detect_query_intents,
    retrieve_knowledge,
)


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"

RAG_GUIDELINE_FILE = (
    KNOWLEDGE_DIR
    / "rag_answer_guideline.txt"
)

DEFAULT_TOP_K = 5

DEFAULT_MODEL = (
    "scb10x/typhoon2.5-qwen3-4b:latest"
)

NO_INFORMATION_MESSAGE = (
    "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"
)


# =========================================================
# System Prompt
# =========================================================

BASE_SYSTEM_PROMPT = """
คุณเป็นผู้ช่วยให้ข้อมูลเกี่ยวกับธาตุเจ้าเรือน
ตามองค์ความรู้การแพทย์แผนไทยที่อยู่ในฐานความรู้ของระบบ

ระบบนี้เน้นผลธาตุเจ้าเรือนปัจจุบันจากแบบประเมิน
และข้อมูลที่ผู้ใช้ระบุในคำถาม

กฎสำคัญ:

1. ตอบโดยใช้เฉพาะข้อมูลที่อยู่ใน CONTEXT เท่านั้น

2. ห้ามใช้ความรู้ทั่วไปของโมเดล
   เพื่อสร้างหรือเติมข้อมูลที่ไม่มีอยู่ใน CONTEXT

3. ห้ามสร้างชื่ออาหาร สมุนไพร เมนู
   หรือข้อควรระวังที่ไม่มีอยู่ใน CONTEXT

4. หากผู้ใช้ถามเกี่ยวกับลักษณะของธาตุ
   ให้ตอบจากข้อมูลประเภท profile

5. หากผู้ใช้ถามเกี่ยวกับอาหาร
   ให้ตอบจากข้อมูลประเภท food

6. หากผู้ใช้ถามเกี่ยวกับเมนู
   ให้ตอบจากข้อมูลประเภท menu

7. หากผู้ใช้ถามเกี่ยวกับข้อควรระวัง
   ให้ตอบจากข้อมูลประเภท warning

8. หากคำถามเกี่ยวข้องกับหลายหัวข้อ
   สามารถใช้ข้อมูลจากหลายประเภทใน CONTEXT ได้

9. หากมีผลธาตุจากแบบประเมิน
   ให้ใช้ผลดังกล่าวเป็นบริบทประกอบคำตอบ

10. หากมีธาตุหลักและธาตุรอง
    ห้ามกล่าวว่าธาตุรองเป็นธาตุหลัก

11. ห้ามวินิจฉัยโรค

12. หากกล่าวถึงอาการหรือความสัมพันธ์กับสุขภาพ
    ให้ใช้ถ้อยคำในลักษณะ
    "มีแนวโน้ม"
    หรือ
    "ตามข้อมูลในฐานความรู้"

13. หาก CONTEXT ไม่มีข้อมูลเพียงพอ
    ให้ตอบว่า:
    "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"

14. ห้ามสร้างแหล่งข้อมูลขึ้นเอง

15. ตอบเป็นภาษาไทย
    อ่านง่าย กระชับ และชัดเจน
""".strip()


# =========================================================
# Load Additional RAG Guideline
# =========================================================

def load_answer_guideline() -> str:
    """
    โหลดกฎเพิ่มเติมสำหรับการสร้างคำตอบ

    ถ้าไม่มีไฟล์ rag_answer_guideline.txt
    ระบบยังสามารถทำงานได้
    """

    if not RAG_GUIDELINE_FILE.exists():
        return ""

    return RAG_GUIDELINE_FILE.read_text(
        encoding="utf-8"
    ).strip()


# =========================================================
# Build System Prompt
# =========================================================

def get_system_prompt() -> str:
    """
    รวม Base System Prompt
    กับ guideline เพิ่มเติม (ถ้ามี)
    """

    guideline = load_answer_guideline()

    if not guideline:
        return BASE_SYSTEM_PROMPT

    return (
        BASE_SYSTEM_PROMPT
        + "\n\n"
        + "=" * 50
        + "\n"
        + "กฎเพิ่มเติมจากฐานความรู้"
        + "\n"
        + "=" * 50
        + "\n\n"
        + guideline
    )


# =========================================================
# Validate Element
# =========================================================

def validate_element(
    element: str | None,
) -> None:
    """
    ตรวจสอบชื่อธาตุ
    """

    if element is None:
        return

    if element not in ELEMENT_NAMES_TH:
        raise ValueError(
            f"Unknown element: {element}"
        )


# =========================================================
# Determine Active Elements
# =========================================================

def determine_active_elements(
    question: str,
    primary_element: str | None = None,
    secondary_element: str | None = None,
) -> list[str]:
    """
    กำหนดธาตุที่จะใช้ Retrieval

    Priority:

    1. ถ้าผู้ใช้ระบุชื่อธาตุในคำถามโดยตรง
       ให้ใช้ธาตุที่ระบุในคำถาม

    2. ถ้าไม่ได้ระบุธาตุในคำถาม
       ให้ใช้ primary / secondary
       จากผลแบบประเมิน

    ตัวอย่าง:

    Question:
        "ธาตุไฟควรกินอะไร"

    Assessment:
        primary = water
        secondary = earth

    Retrieval:
        ["fire"]

    เพราะผู้ใช้ถามธาตุไฟโดยตรง
    """

    validate_element(
        primary_element
    )

    validate_element(
        secondary_element
    )

    explicit_elements = (
        detect_explicit_elements(
            question
        )
    )

    if explicit_elements:
        return explicit_elements

    active_elements: list[str] = []

    if primary_element:

        active_elements.append(
            primary_element
        )

    if (
        secondary_element
        and secondary_element
        != primary_element
    ):

        active_elements.append(
            secondary_element
        )

    return active_elements


# =========================================================
# Determine Recommendation Mode
# =========================================================

def determine_mode(
    active_elements: list[str],
) -> str:
    """
    ระบุรูปแบบธาตุที่ RAG ใช้งาน

    single = ธาตุเดียว
    mixed  = มากกว่า 1 ธาตุ
    unknown = ไม่มีธาตุ
    """

    if not active_elements:
        return "unknown"

    if len(active_elements) == 1:
        return "single"

    return "mixed"


# =========================================================
# Extract Sources
# =========================================================

def extract_sources(
    results: list[dict[str, Any]],
) -> list[str]:
    """
    ดึงรายชื่อไฟล์ต้นทางจาก Retrieval Results
    โดยไม่ให้ชื่อซ้ำ
    """

    sources: list[str] = []

    for item in results:

        source = str(
            item.get(
                "source_file",
                "",
            )
        ).strip()

        if (
            source
            and source not in sources
        ):
            sources.append(
                source
            )

    return sources


# =========================================================
# Create Empty Result
# =========================================================

def create_empty_result(
    question: str,
    primary_element: str | None = None,
    secondary_element: str | None = None,
    active_elements: list[str] | None = None,
) -> dict[str, Any]:
    """
    ผลลัพธ์กรณีไม่พบ Knowledge
    """

    active_elements = (
        active_elements or []
    )

    return {
        "question": question,

        "primary_element": (
            primary_element
        ),

        "secondary_element": (
            secondary_element
        ),

        "active_elements": (
            active_elements
        ),

        "mode": determine_mode(
            active_elements
        ),

        "query_intents": (
            detect_query_intents(
                question
            )
            if question
            else []
        ),

        "answer": (
            NO_INFORMATION_MESSAGE
        ),

        "sources": [],

        "retrieved_chunks": [],
    }


# =========================================================
# Build Assessment Context
# =========================================================

def build_assessment_context(
    primary_element: str | None,
    secondary_element: str | None,
) -> str:
    """
    สร้างข้อความอธิบายผลแบบประเมิน
    เพื่อให้ LLM เข้าใจว่าอะไรคือ
    primary / secondary

    หมายเหตุ:
    ส่วนนี้ไม่ได้เป็น Knowledge Retrieval
    แต่เป็น metadata จากผลแบบประเมิน
    """

    parts: list[str] = []

    if primary_element:

        primary_th = (
            ELEMENT_NAMES_TH[
                primary_element
            ]
        )

        parts.append(
            f"ธาตุหลักจากแบบประเมิน: "
            f"{primary_th}"
        )

    if secondary_element:

        secondary_th = (
            ELEMENT_NAMES_TH[
                secondary_element
            ]
        )

        parts.append(
            f"ธาตุรองจากแบบประเมิน: "
            f"{secondary_th}"
        )

    if not parts:
        return (
            "ไม่มีผลแบบประเมินธาตุ"
        )

    return "\n".join(
        parts
    )


# =========================================================
# Build RAG Prompt
# =========================================================

def build_rag_prompt(
    question: str,
    context: str,
    active_elements: list[str],
    primary_element: str | None = None,
    secondary_element: str | None = None,
) -> str:
    """
    สร้าง User Prompt
    ที่จะส่งให้ LLM
    """

    query_intents = (
        detect_query_intents(
            question
        )
    )

    assessment_context = (
        build_assessment_context(
            primary_element=(
                primary_element
            ),
            secondary_element=(
                secondary_element
            ),
        )
    )

    active_element_names = [
        ELEMENT_NAMES_TH[
            element
        ]
        for element in active_elements
    ]

    active_text = (
        ", ".join(
            active_element_names
        )
        if active_element_names
        else "ไม่ระบุ"
    )

    intent_text = (
        ", ".join(
            query_intents
        )
        if query_intents
        else "general"
    )

    return f"""
คำถามของผู้ใช้
==================================================

{question}


ผลแบบประเมิน
==================================================

{assessment_context}


ธาตุที่ใช้ค้นฐานความรู้
==================================================

{active_text}


ประเภทข้อมูลที่ตรวจพบจากคำถาม
==================================================

{intent_text}


CONTEXT จากฐานความรู้
==================================================

{context}


คำสั่งในการสร้างคำตอบ
==================================================

ตอบคำถามของผู้ใช้โดยใช้เฉพาะข้อมูล
ที่ปรากฏอยู่ใน CONTEXT เท่านั้น

ข้อกำหนด:

- ห้ามใช้ความรู้ทั่วไปของโมเดลมาเติมข้อมูล

- ห้ามสร้างชื่ออาหาร สมุนไพร เมนู
  หรือข้อควรระวังขึ้นเอง

- หากผู้ใช้ถามว่า "ควรกินอะไร"
  ให้ใช้ข้อมูลอาหารจาก CONTEXT

- หากผู้ใช้ถาม "เมนู"
  ให้ใช้ข้อมูลเมนูจาก CONTEXT

- หากผู้ใช้ถาม "ควรหลีกเลี่ยงอะไร"
  ให้ใช้ข้อมูลข้อควรระวังจาก CONTEXT

- หากผู้ใช้ถามหลายเรื่องในคำถามเดียว
  ให้ตอบแยกหัวข้อให้ชัดเจน

- หากมีทั้งธาตุหลักและธาตุรอง
  ให้แยกให้ชัดว่าอะไรคือธาตุหลัก
  และอะไรคือธาตุรอง

- ห้ามกล่าวว่าข้อมูลใน CONTEXT
  เป็นการวินิจฉัยโรค

- ไม่ต้องสร้างชื่อไฟล์หรือแหล่งอ้างอิง
  ในเนื้อหาคำตอบ
  เพราะระบบจะแสดง Sources แยกต่างหาก

- ถ้าข้อมูลใน CONTEXT
  ไม่เพียงพอสำหรับตอบคำถาม
  ให้ตอบเพียง:

"{NO_INFORMATION_MESSAGE}"
""".strip()


# =========================================================
# RAG System
# =========================================================

class RAGSystem:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        RAG System

        Retrieval:
            knowledge_retrieval.py

        Generation:
            Ollama
        """

        if top_k <= 0:

            raise ValueError(
                "top_k ต้องมากกว่า 0"
            )

        self.model = model

        self.top_k = top_k


    # =====================================================
    # Retrieve
    # =====================================================

    def retrieve(
        self,
        question: str,
        active_elements: list[str],
    ) -> list[dict[str, Any]]:
        """
        Retrieve Top-K Knowledge Chunks
        """

        if not active_elements:
            return []

        return retrieve_knowledge(
            query=question,
            active_elements=(
                active_elements
            ),
            top_k=self.top_k,
        )


    # =====================================================
    # Build Context
    # =====================================================

    def build_context(
        self,
        question: str,
        active_elements: list[str],
    ) -> str:
        """
        Build RAG Context
        """

        if not active_elements:
            return ""

        return build_knowledge_context(
            query=question,
            active_elements=(
                active_elements
            ),
            top_k=self.top_k,
        )


    # =====================================================
    # Generate Answer
    # =====================================================

    def generate_answer(
        self,
        question: str,
        context: str,
        active_elements: list[str],
        primary_element: str | None = None,
        secondary_element: str | None = None,
    ) -> str:
        """
        ส่ง Context + Question
        ให้ Ollama สร้างคำตอบ
        """

        if not context.strip():

            return (
                NO_INFORMATION_MESSAGE
            )

        user_prompt = build_rag_prompt(
            question=question,
            context=context,
            active_elements=(
                active_elements
            ),
            primary_element=(
                primary_element
            ),
            secondary_element=(
                secondary_element
            ),
        )

        response = ollama.chat(
            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": (
                        get_system_prompt()
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        user_prompt
                    ),
                },
            ],

            options={
                "temperature": 0.1,
            },
        )

        try:

            answer_text = str(
                response[
                    "message"
                ][
                    "content"
                ]
            ).strip()

        except (
            KeyError,
            TypeError,
        ):

            answer_text = ""

        if not answer_text:

            return (
                NO_INFORMATION_MESSAGE
            )

        return answer_text


    # =====================================================
    # Answer
    # =====================================================

    def answer(
        self,
        question: str,
        primary_element: str | None = None,
        secondary_element: str | None = None,
    ) -> dict[str, Any]:
        """
        Main RAG Pipeline

        Input:
            question
            primary_element
            secondary_element

        Output:
            answer
            sources
            retrieved_chunks
            metadata
        """

        question = question.strip()


        # -------------------------------------------------
        # Empty Question
        # -------------------------------------------------

        if not question:

            return {
                "question": "",

                "primary_element": (
                    primary_element
                ),

                "secondary_element": (
                    secondary_element
                ),

                "active_elements": [],

                "mode": "unknown",

                "query_intents": [],

                "answer": (
                    "กรุณาระบุคำถามก่อนใช้งานระบบ"
                ),

                "sources": [],

                "retrieved_chunks": [],
            }


        # -------------------------------------------------
        # Validate Assessment Elements
        # -------------------------------------------------

        validate_element(
            primary_element
        )

        validate_element(
            secondary_element
        )


        # -------------------------------------------------
        # Determine Active Elements
        # -------------------------------------------------

        active_elements = (
            determine_active_elements(
                question=question,
                primary_element=(
                    primary_element
                ),
                secondary_element=(
                    secondary_element
                ),
            )
        )


        # -------------------------------------------------
        # No Element
        # -------------------------------------------------

        if not active_elements:

            return create_empty_result(
                question=question,
                primary_element=(
                    primary_element
                ),
                secondary_element=(
                    secondary_element
                ),
                active_elements=[],
            )


        # -------------------------------------------------
        # Query Intent
        # -------------------------------------------------

        query_intents = (
            detect_query_intents(
                question
            )
        )


        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        results = self.retrieve(
            question=question,
            active_elements=(
                active_elements
            ),
        )


        if not results:

            return create_empty_result(
                question=question,
                primary_element=(
                    primary_element
                ),
                secondary_element=(
                    secondary_element
                ),
                active_elements=(
                    active_elements
                ),
            )


        # -------------------------------------------------
        # Build Context
        # -------------------------------------------------

        context = self.build_context(
            question=question,
            active_elements=(
                active_elements
            ),
        )


        if not context.strip():

            return create_empty_result(
                question=question,
                primary_element=(
                    primary_element
                ),
                secondary_element=(
                    secondary_element
                ),
                active_elements=(
                    active_elements
                ),
            )


        # -------------------------------------------------
        # Generation
        # -------------------------------------------------

        try:

            answer_text = (
                self.generate_answer(
                    question=question,
                    context=context,
                    active_elements=(
                        active_elements
                    ),
                    primary_element=(
                        primary_element
                    ),
                    secondary_element=(
                        secondary_element
                    ),
                )
            )

        except Exception as error:

            return {
                "question": question,

                "primary_element": (
                    primary_element
                ),

                "secondary_element": (
                    secondary_element
                ),

                "active_elements": (
                    active_elements
                ),

                "mode": determine_mode(
                    active_elements
                ),

                "query_intents": (
                    query_intents
                ),

                "answer": (
                    "ไม่สามารถเชื่อมต่อกับ Ollama ได้"
                    "\n\n"
                    f"รายละเอียด: {error}"
                ),

                "sources": (
                    extract_sources(
                        results
                    )
                ),

                "retrieved_chunks": (
                    results
                ),
            }


        # -------------------------------------------------
        # Final Result
        # -------------------------------------------------

        return {
            "question": question,

            "primary_element": (
                primary_element
            ),

            "secondary_element": (
                secondary_element
            ),

            "active_elements": (
                active_elements
            ),

            "mode": determine_mode(
                active_elements
            ),

            "query_intents": (
                query_intents
            ),

            "answer": (
                answer_text
            ),

            "sources": (
                extract_sources(
                    results
                )
            ),

            "retrieved_chunks": (
                results
            ),
        }


# =========================================================
# Print RAG Result
# =========================================================

def print_rag_result(
    result: dict[str, Any],
) -> None:
    """
    แสดงผลสำหรับ Debug / Manual Test
    """

    print()

    print(
        "=" * 70
    )

    print(
        "คำถาม:",
        result.get(
            "question",
            "",
        ),
    )

    print(
        "Primary Element:",
        result.get(
            "primary_element"
        ),
    )

    print(
        "Secondary Element:",
        result.get(
            "secondary_element"
        ),
    )

    print(
        "Active Elements:",
        ", ".join(
            result.get(
                "active_elements",
                [],
            )
        )
        or "-",
    )

    print(
        "Mode:",
        result.get(
            "mode"
        ),
    )

    print(
        "Query Intents:",
        ", ".join(
            result.get(
                "query_intents",
                [],
            )
        )
        or "-",
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    print()

    print(
        "คำตอบ"
    )

    print(
        "-" * 70
    )

    print(
        result.get(
            "answer",
            "",
        )
    )


    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    print()

    print(
        "แหล่งข้อมูล"
    )

    print(
        "-" * 70
    )

    sources = result.get(
        "sources",
        [],
    )

    if not sources:

        print(
            "ไม่พบแหล่งข้อมูล"
        )

    else:

        for source in sources:

            print(
                f"- {source}"
            )


    # -----------------------------------------------------
    # Retrieval Details
    # -----------------------------------------------------

    retrieved_chunks = (
        result.get(
            "retrieved_chunks",
            [],
        )
    )

    if retrieved_chunks:

        print()

        print(
            "รายละเอียด Knowledge Retrieval"
        )

        print(
            "-" * 70
        )

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):

            print(
                f"{index}. "
                f"{chunk.get('source_file')} "
                f"| element="
                f"{chunk.get('element')} "
                f"| type="
                f"{chunk.get('knowledge_type')} "
                f"| score="
                f"{chunk.get('retrieval_score')}"
            )


    print(
        "=" * 70
    )


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    rag = RAGSystem(
        model=DEFAULT_MODEL,
        top_k=5,
    )


    test_cases = [

        # -------------------------------------------------
        # Explicit Element + Profile
        # -------------------------------------------------

        {
            "question": (
                "ธาตุน้ำมีลักษณะอย่างไร"
            ),

            "primary_element": None,

            "secondary_element": None,
        },


        # -------------------------------------------------
        # Explicit Element + Food
        # -------------------------------------------------

        {
            "question": (
                "ธาตุไฟควรกินอาหารอะไร"
            ),

            "primary_element": None,

            "secondary_element": None,
        },


        # -------------------------------------------------
        # Explicit Element + Menu
        # -------------------------------------------------

        {
            "question": (
                "ธาตุลมมีเมนูอะไรแนะนำ"
            ),

            "primary_element": None,

            "secondary_element": None,
        },


        # -------------------------------------------------
        # Explicit Element + Warning
        # -------------------------------------------------

        {
            "question": (
                "ธาตุน้ำควรกินอะไร "
                "และควรหลีกเลี่ยงอะไร"
            ),

            "primary_element": None,

            "secondary_element": None,
        },


        # -------------------------------------------------
        # Assessment Result
        # No Explicit Element in Question
        # -------------------------------------------------

        {
            "question": (
                "ฉันควรกินอาหารอะไร"
            ),

            "primary_element": (
                "water"
            ),

            "secondary_element": (
                "earth"
            ),
        },


        # -------------------------------------------------
        # Assessment Result + Menu
        # -------------------------------------------------

        {
            "question": (
                "มีเมนูอะไรแนะนำบ้าง"
            ),

            "primary_element": (
                "water"
            ),

            "secondary_element": (
                "earth"
            ),
        },


        # -------------------------------------------------
        # Explicit Element Overrides Assessment Context
        # -------------------------------------------------

        {
            "question": (
                "ธาตุไฟควรหลีกเลี่ยงอะไร"
            ),

            "primary_element": (
                "water"
            ),

            "secondary_element": (
                "earth"
            ),
        },
    ]


    for case in test_cases:

        result = rag.answer(
            question=case[
                "question"
            ],

            primary_element=case[
                "primary_element"
            ],

            secondary_element=case[
                "secondary_element"
            ],
        )

        print_rag_result(
            result
        )

        print()