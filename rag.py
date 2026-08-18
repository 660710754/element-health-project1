from __future__ import annotations

from pathlib import Path
from typing import Any

import ollama

from intent_classifier import classify_question
from recommendation import recommend
from retrieval import TfidfRetriever, build_context


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
RAG_GUIDELINE_FILE = KNOWLEDGE_DIR / "rag_answer_guideline.txt"

DEFAULT_TOP_K = 5
DEFAULT_RECOMMENDATION_LIMIT = 5
MIN_RETRIEVAL_SCORE = 0.05
DEFAULT_MODEL = "scb10x/typhoon2.5-qwen3-4b:latest"

ELEMENT_NAMES_TH = {
    "earth": "ธาตุดิน",
    "water": "ธาตุน้ำ",
    "wind": "ธาตุลม",
    "fire": "ธาตุไฟ",
}

FOOD_CATEGORY_NAMES_TH = {
    "vegetable": "ผัก",
    "fruit": "ผลไม้",
    "menu": "เมนูอาหาร",
    "snack": "อาหารว่าง",
    "drink": "เครื่องดื่ม",
    "warning": "อาหารที่ควรระวัง",
}


BASE_SYSTEM_PROMPT = """
คุณเป็นผู้ช่วยให้ข้อมูลเกี่ยวกับธาตุเจ้าเรือน
ตามองค์ความรู้การแพทย์แผนไทยที่อยู่ในฐานความรู้ของระบบ

ระบบนี้เน้น "ธาตุเจ้าเรือนปัจจุบัน" เป็นหลัก

กฎสำคัญ:
1. ตอบโดยใช้เฉพาะข้อมูลใน CONTEXT เท่านั้น
2. ห้ามสร้างข้อมูลใหม่ที่ไม่มีอยู่ใน CONTEXT
3. ไม่ใช้เดือนเกิดเป็นเหตุผลหลัก เว้นแต่ผู้ใช้ถามโดยตรง
4. หากถามลักษณะของธาตุ ให้ใช้ข้อมูล profile
5. หากถามอาการ/แนวโน้มธาตุ ให้ใช้คำว่า "มีแนวโน้ม" และห้ามวินิจฉัยโรค
6. หากถามอาหาร ให้ตอบเฉพาะหมวดที่ผู้ใช้ถาม
7. หากข้อมูลไม่พอ ให้ตอบว่า "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"
8. หากมีอาการรุนแรงหรือฉุกเฉิน ให้แนะนำพบบุคลากรทางการแพทย์
9. ตอบเป็นภาษาไทย อ่านง่าย ชัดเจน
10. อย่าสร้างรายชื่อแหล่งข้อมูลขึ้นเอง
""".strip()


def load_answer_guideline() -> str:
    if not RAG_GUIDELINE_FILE.exists():
        return ""
    return RAG_GUIDELINE_FILE.read_text(encoding="utf-8").strip()


def get_system_prompt() -> str:
    guideline = load_answer_guideline()
    if not guideline:
        return BASE_SYSTEM_PROMPT

    return (
        BASE_SYSTEM_PROMPT
        + "\n\n==================================================\n"
        + "กฎเพิ่มเติมจาก rag_answer_guideline.txt\n"
        + "==================================================\n\n"
        + guideline
    )


def create_empty_result(
    question: str,
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classification = classification or {}

    return {
        "question": question,
        "intent": classification.get("intent"),
        "element": classification.get("element"),
        "classification": classification,
        "answer": "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ",
        "sources": [],
        "retrieved_chunks": [],
        "recommendation": None,
    }


def extract_sources(
    results: list[dict[str, Any]],
) -> list[str]:
    sources: list[str] = []

    for result in results:
        filename = str(result.get("filename", "")).strip()

        if filename and filename not in sources:
            sources.append(filename)

    return sources


def normalize_source_name(source: str) -> str:
    return source.replace("\\", "/").strip().lower()


def source_matches(
    filename: str,
    expected_source: str,
) -> bool:
    filename_n = normalize_source_name(filename)
    source_n = normalize_source_name(expected_source)

    return (
        filename_n == source_n
        or filename_n.endswith("/" + source_n)
        or Path(filename_n).name == Path(source_n).name
    )


def get_file_priority(
    filename: str,
    classification: dict[str, Any],
) -> int:
    filename_lower = filename.lower()
    intent = str(classification.get("intent", "general"))

    if intent == "profile":
        if "profile" in filename_lower:
            return 0
        if "current_element" in filename_lower:
            return 1
        if "assessment" in filename_lower:
            return 2

    if intent == "symptom":
        if "symptom_element_mapping" in filename_lower:
            return 0
        if "current_element" in filename_lower:
            return 1
        if "assessment" in filename_lower:
            return 2
        if "profile" in filename_lower:
            return 3

    if intent == "element_info":
        if "01_dhatu_theory" in filename_lower:
            return 0
        if "system_overview" in filename_lower:
            return 1
        if "profile" in filename_lower:
            return 2

    if intent == "warning":
        if "warning" in filename_lower:
            return 0

    if intent == "menu":
        if "menu" in filename_lower:
            return 0

    if intent == "food":
        if "food" in filename_lower:
            return 0

    if "rag_answer_guideline" in filename_lower:
        return 8

    if "source_reference" in filename_lower:
        return 9

    return 5


def rerank_results(
    results: list[dict[str, Any]],
    classification: dict[str, Any],
) -> list[dict[str, Any]]:
    target_element = classification.get("element")

    def sort_key(
        item: dict[str, Any],
    ) -> tuple[int, int, float]:
        element = str(item.get("element", "")).lower()
        filename = str(item.get("filename", ""))
        score = float(item.get("score", 0.0))

        if target_element is None:
            element_priority = 0
        elif element == target_element:
            element_priority = 0
        elif element == "system":
            element_priority = 1
        else:
            element_priority = 2

        return (
            element_priority,
            get_file_priority(filename, classification),
            -score,
        )

    return sorted(results, key=sort_key)


def filter_by_classifier_sources(
    results: list[dict[str, Any]],
    classification: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_sources = classification.get("sources", [])

    if not expected_sources:
        return results

    filtered = [
        result
        for result in results
        if any(
            source_matches(
                str(result.get("filename", "")),
                expected_source,
            )
            for expected_source in expected_sources
        )
    ]

    return filtered if filtered else results


def build_scores_for_element(
    element: str,
) -> dict[str, float]:
    if element not in ELEMENT_NAMES_TH:
        raise ValueError(f"ไม่รู้จักธาตุ: {element}")

    scores = {name: 0.0 for name in ELEMENT_NAMES_TH}
    scores[element] = 1.0

    return scores


def recommendation_to_context(
    recommendation_result: dict[str, Any],
) -> str:
    element = recommendation_result.get("element")

    element_th = recommendation_result.get(
        "element_th",
        ELEMENT_NAMES_TH.get(str(element), str(element)),
    )

    intent = recommendation_result.get("intent", "general")
    recommendations = recommendation_result.get("recommendations", {})

    parts = [
        f"ธาตุที่ใช้สำหรับคำแนะนำ: {element_th}",
        f"ประเภทคำถามอาหาร: {intent}",
    ]

    for category, items in recommendations.items():
        if not items:
            continue

        category_th = FOOD_CATEGORY_NAMES_TH.get(category, category)

        parts.append(f"\n[{category_th}]")

        for item in items:
            parts.append(f"- {item}")

    return "\n".join(parts).strip()


def recommendation_sources(
    recommendation_result: dict[str, Any],
) -> list[str]:
    element = str(recommendation_result.get("element", ""))
    recommendations = recommendation_result.get("recommendations", {})

    if not element:
        return []

    sources: list[str] = []

    for category in recommendations.keys():
        if category in {"vegetable", "fruit"}:
            source = f"{element}/{element}_food.txt"

        elif category in {"menu", "snack", "drink"}:
            source = f"{element}/{element}_menu.txt"

        elif category == "warning":
            source = f"{element}/{element}_warning.txt"

        else:
            continue

        if source not in sources:
            sources.append(source)

    return sources


def build_rag_prompt(
    question: str,
    context: str,
    classification: dict[str, Any],
) -> str:
    intent = classification.get("intent", "general")
    element = classification.get("element")

    element_th = (
        ELEMENT_NAMES_TH.get(element, element)
        if element
        else "ไม่ระบุ"
    )

    return f"""
คำถามของผู้ใช้:
{question}

ประเภทคำถามที่ระบบจำแนก:
{intent}

ธาตุที่ตรวจพบในคำถาม:
{element_th}

==================================================
CONTEXT จากฐานความรู้
==================================================

{context}

==================================================
คำสั่งในการตอบ
==================================================

ตอบคำถามโดยอ้างอิงเฉพาะข้อมูลที่อยู่ใน CONTEXT เท่านั้น

ข้อบังคับเพิ่มเติม:
- ห้ามใช้ความรู้ทั่วไปของโมเดลมาเติมคำตอบ
- ห้ามเปลี่ยนแปลงชื่อรายการอาหาร เครื่องดื่ม หรือหมวดหมู่จาก CONTEXT
- หาก CONTEXT มีรายการแบบ bullet ให้ใช้รายการนั้นเป็นหลัก
- หากไม่พบรายการที่ตรงกับคำถามใน CONTEXT ให้ตอบว่า "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"

ถ้าประเภทคำถามเป็น profile:
- ตอบเรื่องลักษณะของธาตุเท่านั้น

ถ้าประเภทคำถามเป็น symptom:
- อธิบายความสัมพันธ์ระหว่างอาการกับแนวโน้มธาตุ
- ห้ามวินิจฉัยโรค
- ใช้คำว่า "มีแนวโน้ม"

ถ้าเป็นคำถามเกี่ยวกับอาหาร:
- ตอบเฉพาะหมวดที่ผู้ใช้ถาม
- ถ้าถามเมนู ให้ตอบเฉพาะเมนู
- ถ้าถามผลไม้ ให้ตอบเฉพาะผลไม้
- ถ้าถามผัก ให้ตอบเฉพาะผัก
- ถ้าถามอาหารว่าง ให้ตอบเฉพาะอาหารว่าง
- ถ้าถามเครื่องดื่ม ให้ตอบเฉพาะเครื่องดื่ม
- ถ้าถามข้อควรระวัง ให้ตอบเฉพาะข้อควรระวัง

ห้ามเพิ่มข้อมูลที่ไม่มีใน CONTEXT

หากข้อมูลไม่เพียงพอ ให้ตอบว่า:
"ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"
""".strip()


class RAGSystem:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
        recommendation_limit: int = DEFAULT_RECOMMENDATION_LIMIT,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k ต้องมากกว่า 0")

        if not 0 <= min_retrieval_score <= 1:
            raise ValueError(
                "min_retrieval_score ต้องอยู่ระหว่าง 0 ถึง 1"
            )

        if recommendation_limit <= 0:
            raise ValueError(
                "recommendation_limit ต้องมากกว่า 0"
            )

        self.model = model
        self.top_k = top_k
        self.min_retrieval_score = min_retrieval_score
        self.recommendation_limit = recommendation_limit
        self.retriever = TfidfRetriever()

    def retrieve(
        self,
        question: str,
        classification: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raw_top_k = max(self.top_k * 4, self.top_k)

        results = self.retriever.search(
            query=question,
            top_k=raw_top_k,
        )

        results = [
            result
            for result in results
            if float(result.get("score", 0.0))
            >= self.min_retrieval_score
        ]

        if not results:
            return []

        results = rerank_results(
            results=results,
            classification=classification,
        )

        results = filter_by_classifier_sources(
            results=results,
            classification=classification,
        )

        final_results = results[: self.top_k]

        for index, result in enumerate(
            final_results,
            start=1,
        ):
            result["rank"] = index

        return final_results

    def build_recommendation_context(
        self,
        question: str,
        classification: dict[str, Any],
        scores: dict[str, float] | None,
    ) -> tuple[str, list[str], dict[str, Any] | None]:
        element = classification.get("element")
        recommendation_scores = scores

        if element is not None:
            recommendation_scores = build_scores_for_element(element)

        if recommendation_scores is None:
            return "", [], None

        result = recommend(
            scores=recommendation_scores,
            question=question,
            limit=self.recommendation_limit,
        )

        context = recommendation_to_context(result)
        sources = recommendation_sources(result)

        return context, sources, result

    def generate_answer(
        self,
        question: str,
        context: str,
        classification: dict[str, Any],
    ) -> str:
        user_prompt = build_rag_prompt(
            question=question,
            context=context,
            classification=classification,
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": get_system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0.2,
            },
        )

        try:
            answer_text = str(
                response["message"]["content"]
            ).strip()

        except (KeyError, TypeError):
            answer_text = ""

        if not answer_text:
            return "ไม่พบข้อมูลเพียงพอในฐานความรู้ของระบบ"

        return answer_text

    def answer(
        self,
        question: str,
        scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        question = question.strip()

        if not question:
            return {
                "question": "",
                "intent": None,
                "element": None,
                "classification": {},
                "answer": "กรุณาระบุคำถามก่อนใช้งานระบบ",
                "sources": [],
                "retrieved_chunks": [],
                "recommendation": None,
            }

        classification = classify_question(question)

        intent = str(
            classification.get(
                "intent",
                "general",
            )
        )

        food_intents = {
            "food",
            "menu",
            "warning",
        }

        if intent in food_intents:
            (
                recommendation_context,
                recommendation_source_list,
                recommendation_result,
            ) = self.build_recommendation_context(
                question=question,
                classification=classification,
                scores=scores,
            )

            if recommendation_context.strip():
                try:
                    answer_text = self.generate_answer(
                        question=question,
                        context=recommendation_context,
                        classification=classification,
                    )

                except Exception as error:
                    return {
                        "question": question,
                        "intent": intent,
                        "element": classification.get("element"),
                        "classification": classification,
                        "answer": (
                            "ไม่สามารถเชื่อมต่อกับ Ollama ได้\n\n"
                            f"รายละเอียด: {error}"
                        ),
                        "sources": recommendation_source_list,
                        "retrieved_chunks": [],
                        "recommendation": recommendation_result,
                    }

                return {
                    "question": question,
                    "intent": intent,
                    "element": (
                        recommendation_result.get("element")
                        if recommendation_result
                        else classification.get("element")
                    ),
                    "classification": classification,
                    "answer": answer_text,
                    "sources": recommendation_source_list,
                    "retrieved_chunks": [],
                    "recommendation": recommendation_result,
                }

        results = self.retrieve(
            question=question,
            classification=classification,
        )

        if not results:
            return create_empty_result(
                question=question,
                classification=classification,
            )

        context = build_context(results)

        if not context.strip():
            return create_empty_result(
                question=question,
                classification=classification,
            )

        try:
            answer_text = self.generate_answer(
                question=question,
                context=context,
                classification=classification,
            )

        except Exception as error:
            return {
                "question": question,
                "intent": intent,
                "element": classification.get("element"),
                "classification": classification,
                "answer": (
                    "ไม่สามารถเชื่อมต่อกับ Ollama ได้\n\n"
                    f"รายละเอียด: {error}"
                ),
                "sources": extract_sources(results),
                "retrieved_chunks": results,
                "recommendation": None,
            }

        return {
            "question": question,
            "intent": intent,
            "element": classification.get("element"),
            "classification": classification,
            "answer": answer_text,
            "sources": extract_sources(results),
            "retrieved_chunks": results,
            "recommendation": None,
        }


def print_rag_result(
    result: dict[str, Any],
) -> None:
    print()
    print("=" * 70)
    print("คำถาม:", result.get("question", ""))
    print("Intent:", result.get("intent"))
    print("Element:", result.get("element"))
    print("=" * 70)

    print()
    print("คำตอบ")
    print("-" * 70)
    print(result.get("answer", ""))

    print()
    print("แหล่งข้อมูล")
    print("-" * 70)

    sources = result.get("sources", [])

    if not sources:
        print("ไม่พบแหล่งข้อมูล")
    else:
        for source in sources:
            print(f"- {source}")

    retrieved_chunks = result.get(
        "retrieved_chunks",
        [],
    )

    if retrieved_chunks:
        print()
        print("รายละเอียด Retrieval")
        print("-" * 70)

        for chunk in retrieved_chunks:
            print(
                f"อันดับ {chunk.get('rank')} | "
                f"{chunk.get('filename')} | "
                f"score={chunk.get('score')}"
            )

    print("=" * 70)


if __name__ == "__main__":
    rag = RAGSystem(
        model=DEFAULT_MODEL,
        top_k=5,
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
        recommendation_limit=5,
    )

    test_cases = [
        {
            "question": "ธาตุเจ้าเรือนปัจจุบันคืออะไร",
            "scores": None,
        },
        {
            "question": "คนธาตุไฟมีลักษณะอย่างไร",
            "scores": None,
        },
        {
            "question": (
                "คนที่ท้องอืด มีลม เวียนหัว "
                "และนอนไม่คงที่ มีแนวโน้มเกี่ยวข้องกับธาตุอะไร"
            ),
            "scores": None,
        },
        {
            "question": "ธาตุไฟควรกินผักอะไร",
            "scores": None,
        },
        {
            "question": "ธาตุลมมีเมนูอะไรบ้าง",
            "scores": None,
        },
        {
            "question": "ธาตุน้ำควรหลีกเลี่ยงอาหารอะไร",
            "scores": None,
        },
        {
            "question": "ฉันควรกินเมนูอะไร",
            "scores": {
                "earth": 12.0,
                "water": 6.0,
                "wind": 4.0,
                "fire": 3.0,
            },
        },
    ]

    for case in test_cases:
        result = rag.answer(
            question=case["question"],
            scores=case["scores"],
        )

        print_rag_result(result)
        print()
