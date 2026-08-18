from __future__ import annotations

from typing import Any


# =========================================================
# Element Mapping
# =========================================================

ELEMENT_KEYWORDS = {

    "earth": [
        "ธาตุดิน",
        "ปถวีธาตุ",
    ],

    "water": [
        "ธาตุน้ำ",
        "อาโปธาตุ",
    ],

    "wind": [
        "ธาตุลม",
        "วาโยธาตุ",
    ],

    "fire": [
        "ธาตุไฟ",
        "เตโชธาตุ",
    ],

}


# =========================================================
# Intent Priority
# สำคัญมาก: ตัวบนมี priority สูงกว่า
# =========================================================

INTENT_PRIORITY = [

    "warning",
    "menu",
    "food",
    "symptom",
    "profile",
    "element_info",

]


# =========================================================
# Intent Keywords
# =========================================================

INTENT_KEYWORDS = {


    "warning": [

        "หลีกเลี่ยง",
        "ไม่ควรกิน",
        "ควรระวัง",
        "ห้ามกิน",
        "อาหารอะไรไม่เหมาะ",
        "ควรงด",

    ],


    "menu": [

        "เมนู",
        "เมนูอาหาร",
        "ทำอาหาร",
        "แนะนำเมนู",

    ],


    "food": [

        "อาหาร",
        "กินอะไร",
        "ควรกินอะไร",
        "กินอะไรดี",
        "ผัก",
        "ผลไม้",

    ],


    "symptom": [

        "อาการ",
        "เป็นอะไร",
        "เกิดจากอะไร",
        "ท้องอืด",
        "มีลม",
        "เวียนหัว",
        "หน้ามืด",
        "ร้อนใน",
        "เหนื่อยง่าย",
        "คัดจมูก",

    ],


    "profile": [

        "ลักษณะ",
        "นิสัย",
        "บุคลิก",
        "รูปร่าง",
        "คนธาตุ",
        "เป็นอย่างไร",
        "ลักษณะของคน",

    ],


    "element_info": [

        "คืออะไร",
        "หมายถึงอะไร",
        "ความหมาย",
        "ธาตุนี้คือ",
        "เกี่ยวกับอะไร",

    ],

}


# =========================================================
# Source Mapping
# =========================================================

def build_sources(
    intent: str,
    element: str | None,
) -> list[str]:

    if intent == "element_info":

        return [
            "01_dhatu_theory.txt"
        ]


    if element is None:

        return []


    if intent == "profile":

        return [
            f"{element}_profile.txt"
        ]


    if intent == "food":

        return [
            f"{element}_food.txt"
        ]


    if intent == "menu":

        return [
            f"{element}_menu.txt"
        ]


    if intent == "warning":

        return [
            f"{element}_warning.txt"
        ]


    if intent == "symptom":

        return [
            "symptom_element_mapping.txt",
            f"{element}_profile.txt",
            f"{element}_warning.txt",
        ]


    return []



# =========================================================
# Detect Element
# =========================================================

def detect_element(
    question: str
) -> tuple[str | None, float]:


    found = []


    for element, keywords in ELEMENT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in question:

                found.append(element)

                break


    if len(found) == 1:

        return found[0], 1.0


    if len(found) > 1:

        return found[0], 0.6


    return None, 0.0



# =========================================================
# Detect Intent
# =========================================================

def detect_intents(
    question: str
) -> list[tuple[str, float]]:


    results = []


    for intent, keywords in INTENT_KEYWORDS.items():

        score = 0


        for keyword in keywords:

            if keyword in question:

                score += 1


        if score > 0:

            confidence = min(
                score / 3,
                1.0
            )


            results.append(
                (
                    intent,
                    round(
                        confidence,
                        2
                    )
                )
            )


    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return results



# =========================================================
# Select Best Intent
# =========================================================

def select_intent(
    intents: list[tuple[str, float]]
) -> tuple[str, float]:


    if not intents:

        return (
            "general",
            0.3
        )


    intent_names = [
        x[0]
        for x in intents
    ]


    # เลือกตาม priority
    for priority in INTENT_PRIORITY:

        if priority in intent_names:

            score = dict(intents)[priority]

            return (
                priority,
                score
            )


    return intents[0]



# =========================================================
# Main Classifier
# =========================================================

def classify_question(
    question: str
) -> dict[str, Any]:


    question = question.strip()


    element, element_confidence = detect_element(
        question
    )


    intents = detect_intents(
        question
    )


    intent, confidence = select_intent(
        intents
    )


    sources = build_sources(
        intent,
        element,
    )


    return {

        "question": question,

        "intent": intent,

        "element": element,

        "element_confidence":
            element_confidence,

        "intent_confidence":
            confidence,

        "sources":
            sources,

        "matched_intents":
            intents,

    }



# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":


    tests = [

        "ธาตุไฟคืออะไร",

        "คนธาตุไฟมีลักษณะอย่างไร",

        "ธาตุไฟควรกินผักอะไร",

        "ธาตุน้ำควรหลีกเลี่ยงอาหารอะไร",

        "ธาตุลมคืออะไร",

        "ธาตุดินท้องอืดควรกินอะไร",

        "ธาตุไฟควรกินเมนูอะไรและระวังอะไร",

    ]


    for q in tests:

        print("=" * 60)

        print(
            q
        )

        print(
            classify_question(q)
        )