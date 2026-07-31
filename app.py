from __future__ import annotations

import csv
from io import BytesIO
from html import escape
from pathlib import Path
from typing import Any

import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scoring import calculate_scores, get_birth_element, get_result
from recommendation import build_recommendation_summary

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
html,body,[class*="css"]{font-family:"Tahoma","Arial",sans-serif;color:var(--dark)}
.stApp{background:linear-gradient(180deg,#fbfcf7 0%,#f2f5e9 55%,#eaf0dd 100%)}
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
div[data-testid="stRadio"] div[role="radiogroup"]{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;align-items:center!important;width:100%!important;min-height:3.5rem;margin:0!important;padding:.55rem 1rem!important;gap:0!important;background-color:var(--light)!important;border:1px solid #d7dec1!important;border-radius:13px!important}
div[data-testid="stRadio"] div[role="radiogroup"]>label{display:flex!important;align-items:center!important;justify-content:center!important;width:100%!important;min-width:0!important;margin:0!important;padding:.55rem .5rem!important;color:#3e481f!important;font-size:.96rem!important;font-weight:650!important;text-align:center!important}
div[data-testid="stRadio"] div[role="radiogroup"] label p{margin:0!important;padding:0!important;text-align:center!important;white-space:nowrap}
hr{border:none;border-top:1px solid #dce2cc;margin:2rem 0}
div[data-testid="stFormSubmitButton"]{width:100%!important;margin-top:1rem} div[data-testid="stFormSubmitButton"]>button{width:100%!important;min-height:3.4rem;border:none;border-radius:13px;background:var(--primary);color:white;font-size:1.08rem;font-weight:800} div[data-testid="stFormSubmitButton"]>button:hover{background:var(--primary-hover);color:white;box-shadow:0 8px 20px rgba(72,83,32,.24)}
div[data-testid="stProgress"]>div>div>div{background-color:var(--primary)}
.result-card{width:100%;box-sizing:border-box;margin:1rem 0 1.5rem;padding:1.8rem 1.5rem;background:linear-gradient(135deg,#485320 0%,#69773a 100%);border-radius:22px;color:white;text-align:center;box-shadow:0 12px 30px rgba(72,83,32,.22)}.result-label{font-size:1rem;opacity:.9;margin-bottom:.35rem}.result-main{font-size:2.1rem;font-weight:850;line-height:1.5;margin-bottom:.8rem}.result-secondary{font-size:1rem;line-height:1.8}
.score-card{width:100%;min-height:150px;box-sizing:border-box;padding:1.25rem .7rem;background:white;border:1px solid var(--border);border-radius:18px;text-align:center;box-shadow:0 7px 22px rgba(72,83,32,.08)}.score-icon{font-size:2rem;margin-bottom:.35rem}.score-name{color:var(--primary);font-size:1.05rem;font-weight:750}.score-number{color:var(--dark);font-size:1.9rem;font-weight:850;margin-top:.25rem}
.safety-note{width:100%;box-sizing:border-box;margin-top:1.7rem;padding:1.1rem 1.35rem;background:var(--warning-bg);border:1px solid #e4d9a8;border-left:6px solid var(--warning-border);border-radius:14px;color:#544b22;font-size:.94rem;line-height:1.8;text-align:left}.safety-note-title{color:#675a24;font-size:1rem;font-weight:800;margin-bottom:.35rem}
.missing-answer{width:100%;box-sizing:border-box;margin-top:1rem;padding:.95rem 1rem;background:#fff3e7;border:1px solid #e9b77f;border-radius:13px;color:#824a17;font-size:.98rem;font-weight:700;text-align:center}
.radar-card{width:100%;box-sizing:border-box;margin:.6rem 0 1.8rem;padding:1rem 1rem .35rem;background:rgba(255,255,255,.96);border:1px solid var(--border);border-radius:20px;box-shadow:0 9px 26px rgba(72,83,32,.08)}
@media(max-width:768px){.block-container{padding:.9rem .75rem 4rem}.main-title{font-size:2rem}div[data-testid="stForm"]{padding:1.2rem .85rem 1.5rem}.question-heading{grid-template-columns:40px minmax(0,1fr);font-size:.94rem}.question-number{width:40px;padding-right:6px}div[data-testid="stRadio"] div[role="radiogroup"]>label{padding-left:.1rem!important;padding-right:.1rem!important;font-size:.82rem!important}div[data-testid="stRadio"] div[role="radiogroup"] label p{white-space:normal}}
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

def render_score_cards(scores: dict[str,float]) -> None:
    for col, element in zip(st.columns(4), ("earth","water","wind","fire")):
        with col:
            st.markdown(f'<div class="score-card"><div class="score-icon">{ELEMENT_ICONS[element]}</div><div class="score-name">{ELEMENT_NAMES[element]}</div><div class="score-number">{scores[element]:.1f}</div></div>', unsafe_allow_html=True)


def render_radar_chart(scores: dict[str, float]) -> None:
    """แสดงคะแนนธาตุทั้ง 4 เป็นเรดาร์ขนาดกะทัดรัด พร้อมรายการคะแนนด้านขวา"""

    element_order = ("earth", "water", "wind", "fire")
    labels = [ELEMENT_NAMES[element] for element in element_order]
    values = [float(scores[element]) for element in element_order]

    angles = [
        index / len(labels) * 2 * math.pi
        for index in range(len(labels))
    ]
    closed_angles = angles + [angles[0]]
    closed_values = values + [values[0]]

    max_score = max(values) if values else 0.0
    axis_max = max(20.0, math.ceil((max_score + 1.0) / 5.0) * 5.0)
    radial_ticks = list(range(5, int(axis_max) + 1, 5))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "Thonburi",
        "Arial Unicode MS",
        "Noto Sans Thai",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # สร้างพื้นที่ 2 ส่วน: กราฟด้านซ้าย และรายการคะแนนด้านขวา
    figure = plt.figure(figsize=(8.2, 3.15))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 0.85],
        wspace=0.18,
    )

    axis = figure.add_subplot(grid[0, 0], polar=True)
    info_axis = figure.add_subplot(grid[0, 1])
    info_axis.axis("off")

    # ธาตุดินอยู่ด้านบน และเรียงตามเข็มนาฬิกา
    axis.set_theta_offset(math.pi / 2)
    axis.set_theta_direction(-1)

    axis.plot(
        closed_angles,
        closed_values,
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        color="#4f74e8",
        zorder=3,
    )
    axis.fill(
        closed_angles,
        closed_values,
        color="#8fa8ff",
        alpha=0.30,
        zorder=2,
    )

    axis.set_xticks(angles)
    axis.set_xticklabels(
        labels,
        fontsize=11,
        fontweight="bold",
        color="#485320",
    )
    axis.tick_params(axis="x", pad=8)

    axis.set_ylim(0, axis_max)
    axis.set_yticks(radial_ticks)
    axis.set_yticklabels(
        [str(tick) for tick in radial_ticks],
        fontsize=8,
        color="#929a7b",
    )
    axis.set_rlabel_position(90)

    axis.grid(color="#dfe4d4", linewidth=0.9)
    axis.spines["polar"].set_color("#d3dac5")
    axis.spines["polar"].set_linewidth(1.0)
    axis.set_facecolor("#ffffff")

    # สีของแต่ละธาตุสำหรับรายการคะแนนด้านขวา
    element_colors = {
        "earth": "#9b6a2f",
        "water": "#36a9dc",
        "wind": "#f4a62a",
        "fire": "#ef5a5a",
    }

    y_positions = [0.78, 0.60, 0.42, 0.24]

    for element, y in zip(element_order, y_positions):
        value = float(scores[element])

        info_axis.scatter(
            0.08,
            y,
            s=115,
            marker="s",
            color=element_colors[element],
            transform=info_axis.transAxes,
        )

        info_axis.text(
            0.16,
            y,
            ELEMENT_NAMES[element],
            fontsize=11,
            color="#485320",
            va="center",
            transform=info_axis.transAxes,
        )

        info_axis.text(
            0.55,
            y,
            f"{value:.1f}",
            fontsize=11,
            fontweight="bold",
            color="#303817",
            va="center",
            transform=info_axis.transAxes,
        )

        info_axis.text(
            0.72,
            y,
            "คะแนน",
            fontsize=9.5,
            color="#77805c",
            va="center",
            transform=info_axis.transAxes,
        )

    figure.patch.set_alpha(0)
    figure.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.05)

    image_buffer = BytesIO()
    figure.savefig(
        image_buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    image_buffer.seek(0)

    # แสดงภาพที่ครอบตัดแล้ว เพื่อไม่ให้มีพื้นที่ว่างจาก canvas ของ Matplotlib
    left_space, chart_column, right_space = st.columns([0.04, 0.92, 0.04])
    with chart_column:
        st.image(image_buffer, use_container_width=True)

    plt.close(figure)


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

def render_food_recommendations(
    recommendation_summary: dict[str, Any],
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

        st.markdown(f"### {title}")

        columns = st.columns(len(foods))

        for column, food in zip(columns, foods):
            with column:
                st.markdown(
                    (
                        '<div class="score-card">'
                        f'<div class="score-name">{escape(food["food_name_th"])}</div>'
                        f'<div style="margin-top:0.5rem;font-size:0.9rem;">'
                        f'{escape(food["recommended_element_th"])}'
                        '</div>'
                        f'<div style="margin-top:0.5rem;font-size:0.85rem;line-height:1.6;">'
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
            st.warning(
                f"{item['food_name_th']} — {item['reason_th']}"
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
        submitted = st.form_submit_button("ประมวลผลแบบประเมิน", use_container_width=True)

    answered = sum(v is not None for v in answers.values())
    st.markdown('<div class="section-header" style="font-size:1.15rem;">ความคืบหน้าในการตอบแบบประเมิน</div>', unsafe_allow_html=True)
    st.progress(answered/total_questions if total_questions else 0)
    st.markdown(f'<div style="text-align:center;color:#66713d;font-size:.92rem;margin-top:.4rem;">ตอบแล้ว {answered} จาก {total_questions} ข้อ</div>', unsafe_allow_html=True)
    render_safety_note()

    if not submitted:
        return
    if birth_month is None:
        st.warning("กรุณาเลือกเดือนเกิดก่อนประมวลผล"); return
    missing = [qid for qid,v in answers.items() if v is None]
    if missing:
        st.markdown(f'<div class="missing-answer">กรุณาตอบคำถามให้ครบทุกข้อ ขณะนี้ยังเหลือ {len(missing)} ข้อ</div>', unsafe_allow_html=True); return

    valid_answers = {qid:int(v) for qid,v in answers.items() if v is not None}
    try:
        scores = calculate_scores(valid_answers, int(birth_month))
    except (ValueError, KeyError) as error:
        st.error(f"ไม่สามารถคำนวณคะแนนได้: {error}"); return

    result = get_result(scores)
    birth_element = get_birth_element(int(birth_month))
    recommendation_summary = build_recommendation_summary(scores)
    st.markdown('<div class="section-header" style="font-size:1.85rem;">ผลการประเมิน</div><div class="section-description">สรุปจากคำตอบของคุณและธาตุเกิดตามเดือนเกิด</div>', unsafe_allow_html=True)
    render_main_result(result, birth_element)
    st.markdown('<div class="section-header">คะแนนธาตุทั้ง 4</div>', unsafe_allow_html=True)
    render_score_cards(scores)

    st.markdown(
        '<div class="section-header" style="margin:1.15rem 0 0.15rem;">'
        'Radar Chart แสดงคะแนนของธาตุเจ้าเรือนปัจจุบันทั้ง 4 ธาตุ'
        '</div>',
        unsafe_allow_html=True,
    )
    render_radar_chart(scores)

    st.markdown('<div class="section-header">ลำดับคะแนน</div>', unsafe_allow_html=True)
    ranking_data = pd.DataFrame([{"ลำดับ":i,"ธาตุ":f"{ELEMENT_ICONS[e]} {ELEMENT_NAMES[e]}","คะแนน":score} for i,(e,score) in enumerate(result["ranking"],1)])
    st.dataframe(ranking_data, use_container_width=True, hide_index=True)
    
    render_food_recommendations(recommendation_summary)

if __name__ == "__main__":
    main()
