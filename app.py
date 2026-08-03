from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from scoring import calculate_scores, get_birth_element, get_result
from recommendation import build_recommendation_summary
from food_asset_catalogue import (
    FoodAssetCatalogue,
    load_food_asset_catalogue,
)

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "data" / "questions.csv"

ELEMENT_NAMES = {"earth": "ธาตุดิน", "water": "ธาตุน้ำ", "wind": "ธาตุลม", "fire": "ธาตุไฟ"}
ELEMENT_ICONS = {"earth": "⛰️", "water": "💧", "wind": "🍃", "fire": "🔥"}
MONTH_NAMES = {1:"มกราคม",2:"กุมภาพันธ์",3:"มีนาคม",4:"เมษายน",5:"พฤษภาคม",6:"มิถุนายน",7:"กรกฎาคม",8:"สิงหาคม",9:"กันยายน",10:"ตุลาคม",11:"พฤศจิกายน",12:"ธันวาคม"}
SECTION_NAMES = {
    "body":"ลักษณะร่างกาย", "temperature":"การตอบสนองต่ออุณหภูมิ",
    "digestion":"ระบบย่อยอาหาร", "elimination":"ระบบขับถ่าย",
    "sleep":"การนอนหลับ", "energy":"พลังงานและความอดทน",
    "activity":"พฤติกรรมและการทำกิจกรรม", "personality":"บุคลิกภาพ",
    "emotion":"อารมณ์และการตอบสนอง", "communication":"การสื่อสาร",
    "cognition":"การเรียนรู้และความคิด", "symptom":"อาการที่เกิดขึ้นเป็นประจำ",
}

st.set_page_config(page_title="แบบประเมินธาตุเจ้าเรือน", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root{--primary:#485320;--primary-hover:#5d6a2b;--dark:#303817;--medium:#65713a;--light:#eef2e2;--border:#d7dec1;--warning-bg:#f5f1df;--warning-border:#9a873f}
html,body,[data-testid="stAppViewContainer"]{font-family:"Tahoma","Arial",sans-serif;color:var(--dark)}
.stApp{background:linear-gradient(180deg,#fbfcf7 0%,#f2f5e9 55%,#eaf0dd 100%);color:var(--dark)}
.block-container{max-width:1100px;padding:2rem 2rem 5rem} #MainMenu,footer{visibility:hidden} header[data-testid="stHeader"]{background-color:transparent}
.main-title{text-align:center;color:var(--primary);font-size:2.7rem;font-weight:800;line-height:1.3;margin:.3rem 0 .5rem}
.main-subtitle{max-width:820px;margin:0 auto 1.5rem;text-align:center;color:var(--medium);font-size:1.05rem;line-height:1.8}
.intro-card{width:100%;box-sizing:border-box;margin:1rem auto 1.8rem;padding:1.5rem 2rem;background:rgba(255,255,255,.96);border:1px solid var(--border);border-radius:20px;box-shadow:0 10px 28px rgba(72,83,32,.09);text-align:center}
.intro-card-title{color:var(--primary);font-size:1.3rem;font-weight:800;margin-bottom:.65rem}.intro-card-text{max-width:780px;margin:0 auto;color:#596436;font-size:1rem;line-height:1.85;text-align:center}
div[data-testid="stForm"]{width:100%;box-sizing:border-box;margin-top:1rem;padding:2rem 2rem 2.2rem;background:rgba(255,255,255,.98);border:1px solid var(--border);border-radius:24px;box-shadow:0 12px 34px rgba(72,83,32,.10)}
.section-header{text-align:center;color:var(--primary);font-size:1.55rem;font-weight:800;line-height:1.5;margin:1.2rem 0 .25rem}.section-description{text-align:center;color:#77805c;font-size:.92rem;line-height:1.6;margin:0 0 1.35rem}
div[data-testid="stSelectbox"]{width:100%} div[data-testid="stSelectbox"] label{display:block;width:100%;text-align:center;color:var(--dark);font-size:1rem;font-weight:700;margin-bottom:.45rem} div[data-baseweb="select"]>div{min-height:3.1rem;background:#f8faf3;border:1px solid #aab58a;border-radius:12px}
.question-heading{display:grid;grid-template-columns:52px minmax(0,1fr);align-items:start;width:100%;box-sizing:border-box;margin-bottom:.6rem;color:var(--dark);font-size:1rem;font-weight:700;line-height:1.7}.question-number{width:52px;box-sizing:border-box;padding-right:8px;text-align:right;color:var(--primary);font-weight:800}.question-text{text-align:left;word-break:break-word}
div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]),div[data-testid="stRadio"],div[data-testid="stRadio"]>div,div[data-testid="stRadio"]>div>div{width:100%!important;max-width:none!important;min-width:100%!important;box-sizing:border-box!important}
div[data-testid="stRadio"]{margin:0 0 1.35rem!important;padding:0!important}
div[data-testid="stRadio"] div[role="radiogroup"]{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;align-items:stretch!important;width:100%!important;min-height:3.5rem;margin:0!important;padding:.45rem!important;gap:.55rem!important;background-color:#f7f9f2!important;border:1px solid #d7dec1!important;border-radius:13px!important}
div[data-testid="stRadio"] div[role="radiogroup"]>label{display:flex!important;align-items:center!important;justify-content:center!important;width:100%!important;min-width:0!important;margin:0!important;padding:.62rem .5rem!important;background:#fff!important;border:1px solid var(--border)!important;border-radius:10px!important;color:#3e481f!important;font-size:.96rem!important;font-weight:700!important;text-align:center!important;cursor:pointer!important;transition:background-color .16s ease,border-color .16s ease,box-shadow .16s ease,transform .16s ease}
div[data-testid="stRadio"] div[role="radiogroup"]>label:hover{background:#f0f4e6!important;border-color:#9eaa7d!important;transform:translateY(-1px)}
div[data-testid="stRadio"] div[role="radiogroup"]>label:has(input:checked){background:var(--primary)!important;border-color:var(--primary)!important;color:#fff!important;box-shadow:0 5px 14px rgba(72,83,32,.20)}
div[data-testid="stRadio"] div[role="radiogroup"]>label:has(input:focus-visible){outline:3px solid rgba(105,119,58,.28)!important;outline-offset:2px}
div[data-testid="stRadio"] div[role="radiogroup"] label p{margin:0!important;padding:0!important;color:#3e481f!important;text-align:center!important;white-space:nowrap}
div[data-testid="stRadio"] div[role="radiogroup"]>label:has(input:checked) p{color:#fff!important}
hr{border:none;border-top:1px solid #dce2cc;margin:2rem 0}
div[data-testid="stFormSubmitButton"]{width:100%!important;margin-top:1rem} div[data-testid="stFormSubmitButton"]>button{width:100%!important;min-height:3.4rem;border:none;border-radius:13px;background:var(--primary);color:white;font-size:1.08rem;font-weight:800} div[data-testid="stFormSubmitButton"]>button:hover{background:var(--primary-hover);color:white;box-shadow:0 8px 20px rgba(72,83,32,.24)}
div[data-testid="stProgress"]>div>div>div{background-color:var(--primary)}
.result-card{width:100%;box-sizing:border-box;margin:1rem 0 1.5rem;padding:1.8rem 1.5rem;background:linear-gradient(135deg,#485320 0%,#69773a 100%);border-radius:22px;color:white;text-align:center;box-shadow:0 12px 30px rgba(72,83,32,.22)}.result-label{font-size:1rem;opacity:.9;margin-bottom:.35rem}.result-main{font-size:2.1rem;font-weight:850;line-height:1.5;margin-bottom:.8rem}.result-secondary{font-size:1rem;line-height:1.8}
.score-card{width:100%;min-height:150px;box-sizing:border-box;padding:1.25rem .7rem;background:white;border:1px solid var(--border);border-radius:18px;text-align:center;box-shadow:0 7px 22px rgba(72,83,32,.08)}.score-icon{font-size:2rem;margin-bottom:.35rem}.score-name{color:var(--primary);font-size:1.05rem;font-weight:750}.score-number{color:var(--dark);font-size:1.9rem;font-weight:850;margin-top:.25rem}
.food-category-title{margin:2.25rem 0 1rem;padding:.15rem 0 .15rem .8rem;border-left:5px solid var(--medium);color:var(--primary);font-size:1.35rem;font-weight:850;line-height:1.4}
[class*="st-key-food-card-"]{height:100%;color:var(--dark)}
[class*="st-key-food-card-"] [data-testid="stVerticalBlockBorderWrapper"]{height:100%;overflow:hidden;background:#fff!important;border:1px solid var(--border)!important;border-radius:20px!important;box-shadow:0 9px 24px rgba(72,83,32,.10)!important}
[class*="st-key-food-card-"] [data-testid="stVerticalBlock"]{gap:.65rem!important}
[class*="st-key-food-card-"] div[data-testid="stImage"]{display:flex!important;align-items:center!important;justify-content:center!important;width:100%!important;margin:0!important}
[class*="st-key-food-card-"] div[data-testid="stImage"] img{display:block;width:100%!important;max-width:100%!important;height:180px!important;margin-inline:auto!important;object-fit:cover!important;object-position:center center!important;border-radius:13px!important}
.food-image-placeholder{display:grid;place-items:center;width:100%;height:180px;border-radius:13px;background:linear-gradient(135deg,#eef2e2,#dfe7cc);color:var(--primary);font-size:2.7rem}
.food-card-copy{display:flex;min-height:168px;box-sizing:border-box;padding:.25rem .35rem .65rem;flex-direction:column;align-items:center;color:var(--dark);text-align:center}
.food-name{color:var(--primary);font-size:1.08rem;font-weight:850;line-height:1.45}
.food-element{display:inline-flex;margin:.6rem 0 .65rem;padding:.28rem .65rem;border-radius:999px;background:var(--light);color:#55612e;font-size:.82rem;font-weight:750;line-height:1.3}
.food-reason{color:#596436;font-size:.88rem;line-height:1.65}
.food-warning-card{display:grid;grid-template-columns:2.35rem minmax(0,1fr);align-items:start;gap:.8rem;width:100%;box-sizing:border-box;margin:.8rem 0;padding:1rem 1.15rem;background:linear-gradient(135deg,#fff9d9 0%,#f6edaa 100%);border:1px solid #dec75b;border-left:6px solid #caa323;border-radius:15px;color:#54440f;box-shadow:0 5px 16px rgba(123,99,18,.08)}
.food-warning-icon{display:grid;place-items:center;width:2.25rem;height:2.25rem;border-radius:50%;background:#f1cf4f;color:#4b3c0d;font-size:1.15rem;line-height:1}
.food-warning-copy{padding-top:.22rem;color:#62511a;font-size:.96rem;line-height:1.7;text-align:left}
.food-warning-name{color:#493b0e;font-weight:850}.food-warning-separator{padding:0 .25rem;color:#8d7524}.food-warning-reason{color:#62511a}
.safety-note{width:100%;box-sizing:border-box;margin-top:1.7rem;padding:1.1rem 1.35rem;background:var(--warning-bg);border:1px solid #e4d9a8;border-left:6px solid var(--warning-border);border-radius:14px;color:#544b22;font-size:.94rem;line-height:1.8;text-align:left}.safety-note-title{color:#675a24;font-size:1rem;font-weight:800;margin-bottom:.35rem}
div[data-testid="stDialog"][role="dialog"],div[data-testid="stDialog"] [role="dialog"]{overflow:hidden;background:#fff!important;border:1px solid var(--border)!important;border-radius:22px!important;box-shadow:0 22px 60px rgba(48,56,23,.28)!important;color:var(--dark)!important}
div[data-testid="stDialog"] [data-testid="stMarkdownContainer"]{color:inherit!important}div[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p{color:inherit!important}
.validation-popup{display:grid;grid-template-columns:3rem minmax(0,1fr);align-items:start;gap:.9rem;width:100%;box-sizing:border-box;margin:.2rem 0 1rem;padding:1.1rem 1.15rem;border-radius:16px;text-align:left}
.validation-popup-warning{background:#fff7cf;border:1px solid #e2c75b;color:#5b4911}.validation-popup-error{background:#fff0ec;border:1px solid #e59a83;color:#762d1d}
.validation-popup-icon{display:grid;place-items:center;width:3rem;height:3rem;border-radius:50%;font-size:1.35rem;line-height:1}.validation-popup-warning .validation-popup-icon{background:#f1cf4f;color:#4b3c0d}.validation-popup-error .validation-popup-icon{background:#d95f43;color:#fff}
.validation-popup-title{margin:.05rem 0 .3rem;font-size:1.05rem;font-weight:850;line-height:1.45}.validation-popup-message{font-size:.94rem;line-height:1.7}.validation-popup-count{font-size:1.08em;font-weight:850}
[class*="st-key-validation-dialog-action-"] button{width:100%!important;min-height:3rem!important;border:none!important;border-radius:12px!important;background:var(--primary)!important;color:#fff!important;font-weight:800!important}[class*="st-key-validation-dialog-action-"] button:hover{background:var(--primary-hover)!important;color:#fff!important}
@media(max-width:768px){.block-container{padding:.9rem .75rem 4rem}.main-title{font-size:2rem}div[data-testid="stForm"]{padding:1.2rem .85rem 1.5rem}.question-heading{grid-template-columns:40px minmax(0,1fr);font-size:.94rem}.question-number{width:40px;padding-right:6px}div[data-testid="stRadio"] div[role="radiogroup"]{gap:.3rem!important;padding:.35rem!important}div[data-testid="stRadio"] div[role="radiogroup"]>label{padding-left:.1rem!important;padding-right:.1rem!important;font-size:.82rem!important}div[data-testid="stRadio"] div[role="radiogroup"] label p{white-space:normal}.food-category-title{margin-top:1.8rem}.food-card-copy{min-height:0}.food-image-placeholder,[class*="st-key-food-card-"] div[data-testid="stImage"] img{height:220px!important}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_questions() -> list[dict[str, str]]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์คำถาม: {QUESTIONS_FILE}")
    with QUESTIONS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    if not questions:
        raise ValueError("questions.csv ไม่มีข้อมูลคำถาม")
    required = {"question_id","section","question_th","response_type","option_0","option_1","option_2","timeframe","is_scored","source_basis"}
    missing = required - set(questions[0].keys())
    if missing:
        raise ValueError("questions.csv ขาดคอลัมน์: " + ", ".join(sorted(missing)))
    return questions

def render_page_header() -> None:
    html = ('<div class="main-title">🌿 แบบประเมินธาตุเจ้าเรือน</div>'
            '<div class="main-subtitle">ประเมินแนวโน้มธาตุดิน ธาตุน้ำ ธาตุลม และธาตุไฟ จากลักษณะร่างกาย พฤติกรรม และอาการที่เกิดขึ้นเป็นประจำ</div>'
            '<div class="intro-card"><div class="intro-card-title">คำแนะนำก่อนเริ่มทำแบบประเมิน</div>'
            '<div class="intro-card-text">โปรดเลือกคำตอบที่ใกล้เคียงกับลักษณะของคุณมากที่สุด<br>สำหรับคำถามเกี่ยวกับอาการ ให้พิจารณาจากสิ่งที่เกิดขึ้นอย่างต่อเนื่องในช่วง 3 เดือนล่าสุด</div></div>')
    st.markdown(html, unsafe_allow_html=True)

def render_section_header(section_name: str, count: int) -> None:
    thai = escape(SECTION_NAMES.get(section_name, section_name))
    st.markdown(f'<div class="section-header">{thai}</div><div class="section-description">จำนวน {count} ข้อ</div>', unsafe_allow_html=True)

def render_question_heading(number: int, text: str) -> None:
    st.markdown(f'<div class="question-heading"><div class="question-number">{number}.</div><div class="question-text">{escape(text)}</div></div>', unsafe_allow_html=True)

def render_safety_note() -> None:
    html = ('<div class="safety-note"><div class="safety-note-title">หมายเหตุด้านสุขภาพ</div>'
            'แบบประเมินนี้ใช้เพื่อคัดกรองแนวโน้มธาตุเจ้าเรือนเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการตรวจหรือการวินิจฉัยโรคจากบุคลากรทางการแพทย์ได้'
            '<br><br>หากคุณมีอาการรุนแรง เช่น เจ็บหน้าอก หายใจลำบาก ไข้สูง อาเจียนมาก ถ่ายดำ ถ่ายเป็นเลือด บวมผิดปกติ หรือน้ำหนักลดโดยไม่ทราบสาเหตุ ควรเข้ารับการตรวจจากแพทย์หรือสถานพยาบาลโดยเร็ว</div>')
    st.markdown(html, unsafe_allow_html=True)

def render_validation_popup(
    *,
    tone: str,
    icon: str,
    title: str,
    message: str,
    button_label: str,
) -> None:
    st.markdown(
        (
            f'<div class="validation-popup validation-popup-{tone}">'
            f'<div class="validation-popup-icon" aria-hidden="true">{icon}</div>'
            '<div>'
            f'<div class="validation-popup-title">{escape(title)}</div>'
            f'<div class="validation-popup-message">{message}</div>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    if st.button(
        button_label,
        key=f"validation-dialog-action-{tone}",
        type="primary",
        width="stretch",
    ):
        st.rerun()

@st.dialog(
    "กรุณาเลือกเดือนเกิด",
    width="small",
    dismissible=True,
    icon="⚠️",
)
def render_birth_month_dialog() -> None:
    render_validation_popup(
        tone="warning",
        icon="📅",
        title="ยังไม่ได้เลือกเดือนเกิด",
        message=(
            "เลือกเดือนเกิดก่อนประมวลผล "
            "เพื่อให้ระบบคำนวณธาตุเกิดเบื้องต้นได้"
        ),
        button_label="กลับไปเลือกเดือนเกิด",
    )

@st.dialog(
    "คำตอบยังไม่ครบ",
    width="small",
    dismissible=True,
    icon="❗",
)
def render_missing_answers_dialog(missing_count: int) -> None:
    render_validation_popup(
        tone="error",
        icon="!",
        title="กรุณาตอบคำถามให้ครบทุกข้อ",
        message=(
            "ขณะนี้ยังเหลือ "
            f'<span class="validation-popup-count">{missing_count}</span> ข้อ '
            "คำตอบที่เลือกไว้จะไม่หาย"
        ),
        button_label="กลับไปตอบต่อ",
    )

def render_score_cards(scores: dict[str,float]) -> None:
    for col, element in zip(st.columns(4), ("earth","water","wind","fire")):
        with col:
            st.markdown(f'<div class="score-card"><div class="score-icon">{ELEMENT_ICONS[element]}</div><div class="score-name">{ELEMENT_NAMES[element]}</div><div class="score-number">{scores[element]:.1f}</div></div>', unsafe_allow_html=True)

def render_main_result(result: dict[str,Any], birth_element: str) -> None:
    primary, secondary = result["primary_element"], result["secondary_element"]
    html = (f'<div class="result-card"><div class="result-label">ธาตุเด่นปัจจุบันของคุณ</div>'
            f'<div class="result-main">{ELEMENT_ICONS[primary]} {ELEMENT_NAMES[primary]}</div>'
            f'<div class="result-secondary">ธาตุเกิดจากเดือนเกิด: <strong>{ELEMENT_NAMES[birth_element]}</strong><br>ธาตุรอง: <strong>{ELEMENT_NAMES[secondary]}</strong></div></div>')
    st.markdown(html, unsafe_allow_html=True)

def group_questions_by_section(questions: list[dict[str,str]]) -> dict[str,list[dict[str,str]]]:
    grouped: dict[str,list[dict[str,str]]] = {}
    for q in questions:
        grouped.setdefault(q["section"].strip(), []).append(q)
    return grouped

@st.cache_resource
def load_assets() -> FoodAssetCatalogue:
    return load_food_asset_catalogue()

def render_food_recommendations(
    recommendation_summary: dict[str, Any],
    catalogue: FoodAssetCatalogue,
) -> None:
    """แสดงอาหารแนะนำและข้อควรระวังบนหน้าเว็บ"""

    st.markdown(
        '<div class="section-header">อาหารที่เหมาะกับแนวโน้มธาตุของคุณ</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="section-description">'
            f'{escape(recommendation_summary["interpretation"])}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    grouped = recommendation_summary["recommendations_by_category"]

    category_titles = {
        "menu": "เมนูอาหาร",
        "vegetable_herb": "ผักพื้นบ้านและสมุนไพร",
        "fruit": "ผลไม้",
        "snack": "อาหารว่าง",
        "drink": "เครื่องดื่ม",
    }

    for category, title in category_titles.items():
        foods = grouped.get(category, [])

        if not foods:
            continue

        st.markdown(
            f'<div class="food-category-title">{escape(title)}</div>',
            unsafe_allow_html=True,
        )

        columns = st.columns(
            len(foods),
            gap="medium",
            vertical_alignment="top",
        )

        for column, food in zip(columns, foods):
            with column:
                matches = catalogue.find_by_name(
                    food["food_name_th"]
                )
                asset = matches[0] if matches else None

                card_key = f'food-card-{food["food_id"]}'

                with st.container(
                    border=True,
                    key=card_key,
                    gap="xsmall",
                ):
                    if asset is not None and asset.is_available:
                        image = asset.images[0]

                        st.image(
                            image.path,
                            width="stretch",
                        )
                    else:
                        st.markdown(
                            '<div class="food-image-placeholder">🌿</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        (
                            '<div class="food-card-copy">'
                            f'<div class="food-name">{escape(food["food_name_th"])}</div>'
                            f'<div class="food-element">'
                            f'{escape(food["recommended_element_th"])}'
                            '</div>'
                            f'<div class="food-reason">'
                            f'{escape(food["match_reason"])}'
                            '</div>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )

    avoid_rules = recommendation_summary["avoid_rules"]

    if avoid_rules:
        st.markdown(
            '<div class="section-header">อาหารที่ควรระวัง</div>',
            unsafe_allow_html=True,
        )

        for item in avoid_rules:
            st.markdown(
                (
                    '<div class="food-warning-card">'
                    '<div class="food-warning-icon" aria-hidden="true">⚠️</div>'
                    '<div class="food-warning-copy">'
                    f'<span class="food-warning-name">{escape(item["food_name_th"])}</span>'
                    '<span class="food-warning-separator" aria-hidden="true">—</span>'
                    f'<span class="food-warning-reason">{escape(item["reason_th"])}</span>'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

def main() -> None:
    render_page_header()
    try:
        questions = load_questions()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error)); st.stop()

    birth_question = next((q for q in questions if q["question_id"].strip()=="Q001"), None)
    if birth_question is None:
        st.error("ไม่พบคำถาม Q001 สำหรับเลือกเดือนเกิด"); st.stop()

    scored_questions = [q for q in questions if q["is_scored"].strip()=="1"]
    grouped = group_questions_by_section(scored_questions)
    total_questions = len(scored_questions)

    with st.form("element_questionnaire"):
        st.markdown('<div class="section-header">ข้อมูลพื้นฐาน</div><div class="section-description">เลือกเดือนเกิดเพื่อใช้คำนวณธาตุเกิดเบื้องต้น</div>', unsafe_allow_html=True)
        birth_month = st.selectbox(birth_question["question_th"].strip(), list(MONTH_NAMES.keys()), format_func=lambda m: MONTH_NAMES[m], index=None, placeholder="กรุณาเลือกเดือนเกิด")
        st.divider()
        answers: dict[str,int|None] = {}
        n = 1
        for section, qs in grouped.items():
            render_section_header(section, len(qs))
            for q in qs:
                qid = q["question_id"].strip()
                options = [q["option_0"].strip(), q["option_1"].strip(), q["option_2"].strip()]
                options = [o for o in options if o]
                render_question_heading(n, q["question_th"].strip())
                selected = st.radio(f"คำตอบข้อ {n}", options, index=None, horizontal=True, key=f"answer_{qid}", label_visibility="collapsed")
                answers[qid] = None if selected is None else options.index(selected)
                n += 1
            st.divider()
        submitted = st.form_submit_button("ประมวลผลแบบประเมิน", width="stretch")

    answered = sum(v is not None for v in answers.values())
    st.markdown('<div class="section-header" style="font-size:1.15rem;">ความคืบหน้าในการตอบแบบประเมิน</div>', unsafe_allow_html=True)
    st.progress(answered/total_questions if total_questions else 0)
    st.markdown(f'<div style="text-align:center;color:#66713d;font-size:.92rem;margin-top:.4rem;">ตอบแล้ว {answered} จาก {total_questions} ข้อ</div>', unsafe_allow_html=True)
    render_safety_note()

    if not submitted:
        return
    if birth_month is None:
        render_birth_month_dialog(); return
    missing = [qid for qid,v in answers.items() if v is None]
    if missing:
        render_missing_answers_dialog(len(missing)); return

    valid_answers = {qid:int(v) for qid,v in answers.items() if v is not None}
    try:
        scores = calculate_scores(valid_answers, int(birth_month))
    except (ValueError, KeyError) as error:
        st.error(f"ไม่สามารถคำนวณคะแนนได้: {error}"); return

    result = get_result(scores)
    birth_element = get_birth_element(int(birth_month))
    recommendation_summary = build_recommendation_summary(scores)
    catalogue = load_assets()

    st.markdown('<div class="section-header" style="font-size:1.85rem;">ผลการประเมิน</div><div class="section-description">สรุปจากคำตอบของคุณและธาตุเกิดตามเดือนเกิด</div>', unsafe_allow_html=True)
    render_main_result(result, birth_element)

    st.markdown('<div class="section-header">คะแนนธาตุทั้ง 4</div>', unsafe_allow_html=True)
    render_score_cards(scores)

    st.markdown('<div class="section-header">กราฟเปรียบเทียบคะแนน</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({"ธาตุ":[ELEMENT_NAMES[e] for e in ("earth","water","wind","fire")], "คะแนน":[scores[e] for e in ("earth","water","wind","fire")]}).set_index("ธาตุ")
    st.bar_chart(chart_data, y="คะแนน", width="stretch")

    st.markdown('<div class="section-header">ลำดับคะแนน</div>', unsafe_allow_html=True)
    ranking_data = pd.DataFrame([{"ลำดับ":i,"ธาตุ":f"{ELEMENT_ICONS[e]} {ELEMENT_NAMES[e]}","คะแนน":score} for i,(e,score) in enumerate(result["ranking"],1)])
    st.dataframe(ranking_data, width="stretch", hide_index=True)

    render_food_recommendations(
        recommendation_summary,
        catalogue,
    )

if __name__ == "__main__":
    main()
