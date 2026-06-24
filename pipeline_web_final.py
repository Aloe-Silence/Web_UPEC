"""
Web-enabled dynamic Bayesian network assessment of underground pipeline
external corrosion.

Run:
    streamlit run pipeline_web_final.py
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pipeline_dbn_service import PipelineCorrosionDBN


MODEL_PATH = Path(__file__).with_name("DBN-GZ.xdsl")
TIME_STEP_YEARS = 2
AGE_THRESHOLD_YEARS = 30
PREDICTION_SLICES = 13
TARGET_NODE = "E1"
TARGET_STATES = ("Low", "Moderate", "High")

NODE_LABELS = {
    "X1": "pH value",
    "X2": "Soil resistivity",
    "X3": "Redox potential",
    "X4": "Dissolved chloride ion concentration",
    "X5": "Sulfate ion concentration",
    "X6": "Bicarbonate ion concentration",
    "X7": "Bulk density",
    "X8": "Water content",
    "X9": "Soil type",
    "X10": "Coating type",
    "X11": "Pipeline age",
    "X12": "Construction vibration",
    "X13": "Agricultural activity",
    "X14": "Excavate frequency",
    "X15": "Pipe-to-soil potential",
    "X16": "Stray current",
    "X17": "Extreme weather",
    "X18": "Ground movement",
    "X19": "Seismic risk",
    "X20": "Flooding risk",
}

# The original GeNIe model uses a different internal ordering for X1-X9 than
# the paper. Keep the public interface aligned with the paper while preserving
# the original XDSL probability tables and dynamic inference behavior.
MODEL_INPUT_NODE_IDS = {
    "X1": "X6",   # pH value
    "X2": "X9",   # Soil resistivity
    "X3": "X4",   # Redox potential
    "X4": "X7",   # Dissolved chloride ion concentration
    "X5": "X5",   # Sulfate ion concentration
    "X6": "X8",   # Bicarbonate ion concentration
    "X7": "X2",   # Bulk density
    "X8": "X3",   # Water content
    "X9": "X1",   # Soil type
    **{f"X{index}": f"X{index}" for index in range(10, 21)},
}

NODE_GROUPS = (
    ("Physical index", ("X9", "X7", "X8")),
    ("Electrochemical index", ("X2", "X1", "X3")),
    ("Ionic index", ("X4", "X6", "X5")),
    ("Coating failure", ("X10",)),
    ("Third-party activity", ("X12", "X13", "X14")),
    ("Stray current corrosion", ("X15", "X16")),
    ("Operational stress", ("X17", "X18", "X19", "X20")),
)

# Baseline evidence supplied for the 2007 case-study configuration.
DEFAULT_STATES = {
    "X1": "Moderate",
    "X2": "Low",
    "X3": "Low",
    "X4": "Moderate",
    "X5": "Moderate",
    "X6": "Moderate",
    "X7": "Great",
    "X8": "High",
    "X9": "C",
    "X10": "CTC",
    "X12": "Low",
    "X13": "Present",
    "X14": "Low",
    "X15": "High",
    "X16": "Low",
    "X17": "Occurrence",
    "X18": "Nonoccurrence",
    "X19": "Nonoccurrence",
    "X20": "Occurrence",
}

STATE_STYLE = {
    "Low": {"color": "#0072B2", "symbol": "circle"},
    "Moderate": {"color": "#E69F00", "symbol": "diamond"},
    "High": {"color": "#D55E00", "symbol": "square"},
}

SCENARIO_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)

SVG_ICONS = {
    "input": (
        '<path d="M4 21v-7"/><path d="M4 10V3"/>'
        '<path d="M12 21v-9"/><path d="M12 8V3"/>'
        '<path d="M20 21v-5"/><path d="M20 12V3"/>'
        '<path d="M2 14h4"/><path d="M10 8h4"/><path d="M18 16h4"/>'
    ),
    "function": (
        '<rect x="3" y="3" width="7" height="7" rx="1"/>'
        '<rect x="14" y="3" width="7" height="7" rx="1"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1"/>'
        '<rect x="14" y="14" width="7" height="7" rx="1"/>'
    ),
    "analysis": (
        '<path d="M3 3v18h18"/><path d="m7 16 4-4 3 3 5-7"/>'
        '<path d="M17 8h2v2"/>'
    ),
    "output": (
        '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/>'
        '<path d="M5 21h14a2 2 0 0 0 2-2v-3"/>'
        '<path d="M3 16v3a2 2 0 0 0 2 2"/>'
    ),
    "scenario": (
        '<circle cx="6" cy="5" r="2"/><circle cx="18" cy="8" r="2"/>'
        '<circle cx="8" cy="19" r="2"/>'
        '<path d="M6 7v5a7 7 0 0 0 7 7h3"/>'
        '<path d="M8 5h3a7 7 0 0 1 7 7v2"/>'
    ),
    "model": (
        '<circle cx="12" cy="4" r="2"/><circle cx="5" cy="18" r="2"/>'
        '<circle cx="19" cy="18" r="2"/><circle cx="12" cy="13" r="2"/>'
        '<path d="m11 6-4.5 10"/><path d="m13 6 4.5 10"/>'
        '<path d="M7 18h10"/><path d="m12 6v5"/>'
    ),
    "clock": (
        '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>'
        '<circle cx="12" cy="12" r="1"/>'
    ),
}


def icon_svg(name, css_class="ui-icon"):
    return (
        f'<svg class="{css_class}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{SVG_ICONS[name]}</svg>'
    )


def pipeline_hero_svg():
    return """
        <svg class="pipeline-hero-icon" viewBox="0 0 190 92" fill="none" aria-hidden="true">
            <path d="M8 68 C42 57, 72 77, 106 65 S158 52, 182 62" stroke="rgba(210,240,239,0.34)" stroke-width="2"/>
            <path d="M15 45 H64 V58 H126 V35 H176" stroke="#C7F0EC" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M15 45 H64 V58 H126 V35 H176" stroke="#2D8D96" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="64" cy="45" r="12" fill="#0D5577" stroke="#E4FAF7" stroke-width="2"/>
            <path d="M56 37 L72 53 M72 37 L56 53" stroke="#E4FAF7" stroke-width="2"/>
            <circle cx="126" cy="58" r="6" fill="#D77A46" stroke="#FFE0C9" stroke-width="2"/>
            <circle cx="145" cy="35" r="3" fill="#F1B46F"/>
            <circle cx="154" cy="35" r="2" fill="#F1B46F"/>
            <path d="M94 18 v18 M86 27 h16" stroke="#BFE8E5" stroke-width="2" stroke-linecap="round"/>
            <path d="M80 78 h30" stroke="rgba(255,255,255,0.38)" stroke-width="2" stroke-linecap="round"/>
        </svg>
    """


def module_header_html(module_code, title, subtitle, icon):
    return f"""
        <div class="module-banner" data-module="{module_code}">
            <div class="module-icon-box">{icon_svg(icon, "module-icon")}</div>
            <div class="module-copy">
                <div class="module-code">{module_code}</div>
                <div class="module-title">{title}</div>
                <div class="module-subtitle">{subtitle}</div>
            </div>
        </div>
    """


st.set_page_config(
    page_title="UPEC Dynamic Risk Assessment",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #142534;
            --muted: #60717f;
            --rule: #d3dee5;
            --paper: #f4f7f9;
            --wash: #eaf1f5;
            --accent: #0c5577;
            --accent-2: #c86538;
            --teal: #0c7b7d;
            --tsw: #f1d59b;
            --shadow-sm: 0 4px 14px rgba(23, 52, 70, 0.07);
            --shadow-md: 0 14px 36px rgba(18, 48, 67, 0.12);
        }

        html, body, [class*="css"] {
            font-family: Arial, Helvetica, sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 85% 8%, rgba(12, 123, 125, 0.08), transparent 27rem),
                radial-gradient(circle at 12% 35%, rgba(200, 101, 56, 0.06), transparent 24rem),
                linear-gradient(180deg, #f8fafb 0%, var(--paper) 100%);
        }
        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: rgba(255,255,255,0.96); }

        .block-container {
            max-width: 100% !important;
            min-height: calc(100vh - 2.5rem);
            width: 100% !important;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: clamp(1rem, 2.2vw, 2.75rem);
            padding-right: clamp(1rem, 2.2vw, 2.75rem);
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.72), rgba(234,241,245,0.96)),
                var(--wash);
            border-right: 1px solid var(--rule);
            box-shadow: 8px 0 26px rgba(24, 54, 72, 0.06);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem;
        }

        .paper-header {
            background:
                radial-gradient(circle at 91% 14%, rgba(89, 199, 193, 0.25), transparent 15rem),
                radial-gradient(circle at 70% 115%, rgba(217, 121, 74, 0.23), transparent 22rem),
                linear-gradient(125deg, #082d47 0%, #0b4b68 57%, #0b666f 100%);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            color: #ffffff;
            margin-bottom: 1.45rem;
            overflow: hidden;
            padding: 1.65rem 1.75rem 1.45rem 1.75rem;
            position: relative;
        }

        .paper-header::after {
            border: 1px solid rgba(255,255,255,0.13);
            border-radius: 50%;
            content: "";
            height: 210px;
            position: absolute;
            right: -78px;
            top: -92px;
            width: 210px;
        }

        .paper-header.compact {
            margin-bottom: 1rem;
            padding: 1rem 1.35rem 0.95rem 1.35rem;
        }

        .paper-header.compact .paper-title { font-size: 1.55rem; }
        .paper-header.compact .paper-subtitle { display: none; }
        .paper-header.compact .paper-authors { margin-top: 0.4rem; }
        .paper-header.compact .paper-network-mark { height: 58px; min-width: 64px; width: 64px; }
        .paper-header.compact .header-icon { height: 30px; width: 30px; }
        .paper-header.compact .pipeline-hero-icon { height: 48px; width: 56px; }
        .paper-header.compact .metadata-row { margin-top: 0.55rem; }

        .paper-heading-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 2rem;
        }

        .paper-heading-copy { max-width: 1180px; position: relative; z-index: 1; }

        .paper-network-mark {
            align-items: center;
            backdrop-filter: blur(8px);
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.28);
            border-radius: 12px;
            color: #bce8e6;
            display: flex;
            height: 92px;
            justify-content: center;
            min-width: 190px;
            position: relative;
            width: 190px;
            z-index: 1;
        }

        .header-icon { height: 36px; width: 36px; }
        .pipeline-hero-icon { height: 82px; width: 176px; }

        .paper-kicker {
            color: #9edbd9;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .paper-title {
            color: #ffffff;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2rem;
            font-weight: 600;
            line-height: 1.2;
            margin-bottom: 0.45rem;
        }

        .paper-subtitle {
            color: rgba(239, 248, 251, 0.86);
            font-size: 0.89rem;
            line-height: 1.55;
        }

        .paper-authors {
            color: #ffffff;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 0.96rem;
            font-weight: 600;
            margin-top: 0.72rem;
        }

        .paper-affiliation {
            color: rgba(232, 243, 247, 0.76);
            font-size: 0.8rem;
            line-height: 1.45;
            margin-top: 0.18rem;
        }

        .metadata-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.8rem;
            position: relative;
            z-index: 1;
        }

        .metadata-item {
            align-items: center;
            backdrop-filter: blur(6px);
            background: rgba(255,255,255,0.09);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 999px;
            color: rgba(250,253,255,0.92);
            display: inline-flex;
            font-size: 0.76rem;
            gap: 0.35rem;
            padding: 0.34rem 0.62rem;
        }

        .metadata-icon { height: 14px; width: 14px; }

        .module-banner {
            align-items: center;
            background:
                linear-gradient(110deg, rgba(255,255,255,0.98), rgba(243,248,250,0.96));
            border: 1px solid var(--rule);
            border-left: 4px solid var(--accent);
            border-radius: 10px;
            box-shadow: var(--shadow-sm);
            display: flex;
            gap: 0.78rem;
            margin: 0.8rem 0 1rem 0;
            padding: 0.72rem 0.85rem;
            position: relative;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            overflow: hidden;
        }

        .module-banner::after {
            color: rgba(12, 85, 119, 0.045);
            content: attr(data-module);
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            position: absolute;
            right: 1rem;
            top: 50%;
            transform: translateY(-50%);
            white-space: nowrap;
        }

        .module-banner:hover {
            border-color: #b9cbd6;
            box-shadow: 0 9px 24px rgba(19, 55, 77, 0.11);
            transform: translateY(-1px);
        }

        .dashboard-intro {
            margin: 1.25rem 0 1.1rem 0;
            text-align: center;
        }

        .dashboard-eyebrow {
            color: var(--accent-2);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .dashboard-title {
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.7rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }

        .dashboard-lead {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
            margin: 0.4rem auto 0 auto;
            max-width: 760px;
        }

        .dashboard-grid {
            display: grid;
            gap: 1.1rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin: 1.2rem 0 1.5rem 0;
        }

        .dashboard-card {
            background: linear-gradient(150deg, rgba(255,255,255,0.98), rgba(242,248,250,0.96));
            border: 1px solid #d0dde5;
            border-radius: 16px;
            box-shadow: 0 10px 28px rgba(18, 51, 70, 0.09);
            color: var(--ink) !important;
            display: flex;
            flex-direction: column;
            min-height: clamp(310px, 34vh, 390px);
            overflow: hidden;
            padding: 1.25rem;
            position: relative;
            text-decoration: none !important;
            transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
        }

        .dashboard-card::before {
            background: linear-gradient(90deg, var(--accent), var(--teal));
            content: "";
            height: 5px;
            left: 0;
            position: absolute;
            right: 0;
            top: 0;
        }

        .dashboard-card.scenario::before {
            background: linear-gradient(90deg, var(--accent-2), #e5a34c);
        }

        .dashboard-card.method::before {
            background: linear-gradient(90deg, #594a8a, #198a92);
        }

        .dashboard-card:hover {
            border-color: #a9c1cf;
            box-shadow: 0 18px 42px rgba(15, 50, 71, 0.16);
            transform: translateY(-6px);
        }

        .dashboard-card-top {
            align-items: center;
            display: flex;
            justify-content: space-between;
        }

        .dashboard-card-icon {
            align-items: center;
            background: linear-gradient(145deg, var(--accent), var(--teal));
            border-radius: 13px;
            box-shadow: 0 8px 18px rgba(12,85,119,0.24);
            color: #ffffff;
            display: flex;
            height: 52px;
            justify-content: center;
            width: 52px;
        }

        .dashboard-card.scenario .dashboard-card-icon {
            background: linear-gradient(145deg, var(--accent-2), #dc9945);
        }

        .dashboard-card.method .dashboard-card-icon {
            background: linear-gradient(145deg, #594a8a, #198a92);
        }

        .dashboard-card-icon svg { height: 27px; width: 27px; }

        .dashboard-card-number {
            color: rgba(12,85,119,0.13);
            font-size: 2.3rem;
            font-weight: 800;
        }

        .dashboard-card-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.28rem;
            font-weight: 600;
            margin-top: 1rem;
        }

        .dashboard-card-text {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.55;
            margin-top: 0.48rem;
            min-height: 3.8rem;
        }

        .dashboard-card-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.9rem;
        }

        .dashboard-pill {
            background: #edf4f7;
            border: 1px solid #d5e2e8;
            border-radius: 999px;
            color: #49616f;
            font-size: 0.68rem;
            padding: 0.24rem 0.48rem;
        }

        .dashboard-card-action {
            color: var(--accent);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin-top: auto;
            padding-top: 1rem;
            text-transform: uppercase;
        }

        .workflow-rail {
            align-items: stretch;
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--rule);
            border-radius: 14px;
            box-shadow: var(--shadow-sm);
            display: grid;
            gap: 0;
            grid-template-columns: repeat(4, 1fr);
            margin: 1rem 0 1.4rem 0;
            overflow: hidden;
        }

        .workflow-step {
            border-right: 1px solid var(--rule);
            padding: 0.85rem 0.9rem;
            text-align: center;
        }

        .workflow-step:last-child { border-right: none; }
        .workflow-step strong { color: var(--accent); display: block; font-size: 0.76rem; }
        .workflow-step span { color: var(--muted); font-size: 0.68rem; }

        .back-link {
            align-items: center;
            background: rgba(255,255,255,0.86);
            border: 1px solid #cfdae1;
            border-radius: 999px;
            color: var(--accent) !important;
            display: inline-flex;
            font-size: 0.74rem;
            font-weight: 800;
            gap: 0.35rem;
            margin: 0 0 0.75rem 0;
            padding: 0.42rem 0.72rem;
            text-decoration: none !important;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .back-link:hover { box-shadow: var(--shadow-sm); transform: translateX(-2px); }

        .method-card-grid {
            display: grid;
            gap: 0.9rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.8rem 0 1.2rem 0;
        }

        .method-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--rule);
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            padding: 1rem;
        }

        .method-card strong {
            color: var(--accent);
            display: block;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1rem;
            margin-bottom: 0.3rem;
        }

        .method-card span { color: var(--muted); font-size: 0.76rem; line-height: 1.5; }

        .module-icon-box {
            align-items: center;
            background: linear-gradient(145deg, var(--accent), var(--teal));
            border-radius: 9px;
            box-shadow: 0 5px 12px rgba(12, 85, 119, 0.22);
            color: #ffffff;
            display: flex;
            height: 38px;
            justify-content: center;
            min-width: 38px;
            width: 38px;
        }

        .module-icon { height: 21px; width: 21px; }
        .module-copy { min-width: 0; position: relative; z-index: 1; }

        .module-code {
            color: #6c7b86;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            line-height: 1.1;
            text-transform: uppercase;
        }

        .module-title {
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.02rem;
            font-weight: 600;
            line-height: 1.25;
            margin-top: 0.12rem;
        }

        .module-subtitle {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.08rem;
        }

        section[data-testid="stSidebar"] .module-banner {
            background: #ffffff;
            border-left-color: var(--accent-2);
            margin: 0 0 0.85rem 0;
            padding: 0.64rem 0.68rem;
        }

        section[data-testid="stSidebar"] .module-banner::after { display: none; }

        section[data-testid="stSidebar"] .module-subtitle {
            font-size: 0.72rem;
        }

        .section-heading {
            border-bottom: 1px solid var(--rule);
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.18rem;
            font-weight: 600;
            margin: 0.65rem 0 0.7rem 0;
            padding-bottom: 0.42rem;
        }

        .method-note {
            background: linear-gradient(105deg, #f1f7fa, #f8fbfc);
            border: 1px solid #d9e5eb;
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            color: #34424e;
            font-size: 0.86rem;
            line-height: 1.55;
            margin: 0.35rem 0 1rem 0;
            padding: 0.82rem 1rem;
            box-shadow: 0 3px 10px rgba(24, 61, 82, 0.05);
        }

        .figure-caption {
            color: #4b5660;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 0.84rem;
            line-height: 1.45;
            margin: -0.25rem 0 1.15rem 0;
            padding-left: 0.15rem;
        }

        .sidebar-title {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin: 0.15rem 0 0.6rem 0;
        }

        .sidebar-note {
            color: var(--muted);
            font-size: 0.77rem;
            line-height: 1.45;
            margin-bottom: 0.45rem;
        }

        button[data-baseweb="tab"] {
            color: #45515c;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.035em;
            border-radius: 999px;
            margin: 0.22rem 0.22rem 0.32rem 0;
            padding-left: 1rem;
            padding-right: 1rem;
            transition: background-color 160ms ease, color 160ms ease;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(105deg, var(--accent), var(--teal));
            color: #ffffff !important;
            border-bottom-color: transparent !important;
            box-shadow: 0 5px 12px rgba(12, 85, 119, 0.2);
        }

        div[data-testid="stMetric"] {
            background:
                linear-gradient(145deg, #ffffff 0%, #f7fafb 100%);
            border: 1px solid #d6e0e6;
            border-radius: 10px;
            box-shadow: var(--shadow-sm);
            overflow: hidden;
            padding: 0.82rem 0.92rem;
            position: relative;
            transition: transform 180ms ease, box-shadow 180ms ease;
        }

        div[data-testid="stMetric"]::before {
            background: linear-gradient(90deg, var(--accent), var(--teal), var(--accent-2));
            content: "";
            height: 3px;
            left: 0;
            position: absolute;
            right: 0;
            top: 0;
        }

        div[data-testid="stMetric"]:hover {
            box-shadow: 0 10px 24px rgba(20, 58, 80, 0.12);
            transform: translateY(-2px);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.76rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--rule);
            border-radius: 10px;
            box-shadow: var(--shadow-sm);
            overflow: hidden;
        }

        div[data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.9);
            border: 1px solid #d8e2e8;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            margin: 0.35rem 0 0.7rem 0;
            overflow: hidden;
            padding: 0.35rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 9px;
            box-shadow: 0 4px 14px rgba(22, 61, 84, 0.07);
        }

        div[data-testid="stDownloadButton"] button,
        div[data-testid="stButton"] button {
            background: linear-gradient(105deg, var(--accent), var(--teal));
            border: none;
            border-radius: 8px;
            box-shadow: 0 5px 12px rgba(12, 85, 119, 0.18);
            color: #ffffff;
            font-weight: 700;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stButton"] button:hover {
            box-shadow: 0 8px 18px rgba(12, 85, 119, 0.27);
            color: #ffffff;
            transform: translateY(-1px);
        }

        section[data-testid="stSidebar"] details {
            background: rgba(255,255,255,0.68);
            border: 1px solid #d4e0e6;
            border-radius: 9px;
            box-shadow: 0 3px 10px rgba(21, 57, 78, 0.045);
            margin-bottom: 0.58rem;
            overflow: hidden;
        }

        div[data-testid="stExpander"] details {
            background: rgba(255,255,255,0.88);
            border: 1px solid #cfdae2;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 0.85rem;
            overflow: hidden;
            transition: border-color 160ms ease, box-shadow 160ms ease;
        }

        div[data-testid="stExpander"] details:hover {
            border-color: #afc4d0;
            box-shadow: 0 9px 22px rgba(18, 53, 73, 0.1);
        }

        div[data-testid="stExpander"] summary {
            background: linear-gradient(100deg, rgba(246,250,251,0.98), rgba(237,245,248,0.92));
            font-weight: 700;
            min-height: 3.4rem;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input {
            border-radius: 8px;
        }

        @keyframes fade-up {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .paper-header { animation: fade-up 420ms ease-out both; }
        .module-banner { animation: fade-up 360ms ease-out both; }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }

        @media (max-width: 900px) {
            .paper-network-mark { display: none; }
            .paper-title { font-size: 1.65rem; }
            .paper-header { padding: 1.25rem; }
            .metadata-item { border-radius: 7px; }
            .dashboard-grid, .method-card-grid { grid-template-columns: 1fr; }
            .workflow-rail { grid-template-columns: repeat(2, 1fr); }
            .workflow-step:nth-child(2) { border-right: none; }
            .workflow-step:nth-child(-n+2) { border-bottom: 1px solid var(--rule); }
        }

        .app-footer {
            background: linear-gradient(105deg, rgba(8,45,71,0.97), rgba(11,102,111,0.95));
            border-radius: 10px;
            color: rgba(247,251,253,0.84);
            font-size: 0.76rem;
            line-height: 1.5;
            margin-top: 2.4rem;
            padding: 1rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
        }

        div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
            min-height: calc(100vh - 5.5rem);
        }

        div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"]
        > div[data-testid="stElementContainer"]:has(.app-footer) {
            margin-top: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_model():
    """Keep a separate mutable network instance in each Streamlit session."""
    model_path = str(MODEL_PATH.resolve())
    if (
        "pipeline_dbn_model" not in st.session_state
        or st.session_state.get("pipeline_dbn_model_path") != model_path
    ):
        st.session_state.pipeline_dbn_model = PipelineCorrosionDBN(str(MODEL_PATH))
        st.session_state.pipeline_dbn_model.net.set_slice_count(PREDICTION_SLICES)
        st.session_state.pipeline_dbn_model_path = model_path
        st.session_state.pipeline_dbn_max_slices = PREDICTION_SLICES
    if "pipeline_dbn_max_slices" not in st.session_state:
        st.session_state.pipeline_dbn_max_slices = PREDICTION_SLICES
    return st.session_state.pipeline_dbn_model


def get_model_states(model, node_id):
    return [
        model.net.get_outcome_id(node_id, index)
        for index in range(model.net.get_outcome_count(node_id))
    ]


def get_input_states(model, paper_node_id):
    return get_model_states(model, MODEL_INPUT_NODE_IDS[paper_node_id])


def validate_model_contract(model):
    for node_id in NODE_LABELS:
        if not get_input_states(model, node_id):
            raise ValueError(f"Model node {node_id} has no states.")

    target_states = tuple(get_model_states(model, TARGET_NODE))
    if target_states != TARGET_STATES:
        raise ValueError(
            f"{TARGET_NODE} states are {target_states}; expected {TARGET_STATES}."
        )


def pipeline_age_state(age_years):
    return "Young" if age_years <= AGE_THRESHOLD_YEARS else "Old"


def build_age_schedule(current_age, slice_count):
    return {
        slice_index: pipeline_age_state(
            current_age + slice_index * TIME_STEP_YEARS
        )
        for slice_index in range(slice_count)
    }


def format_age_schedule(schedule, start_year):
    states = list(schedule.values())
    end_year = start_year + (len(states) - 1) * TIME_STEP_YEARS
    if len(set(states)) == 1:
        return f"{states[0]} throughout ({start_year}-{end_year})"

    first_old_slice = states.index("Old")
    last_young_year = start_year + (first_old_slice - 1) * TIME_STEP_YEARS
    first_old_year = start_year + first_old_slice * TIME_STEP_YEARS
    return (
        f"Young: {start_year}-{last_young_year}; "
        f"Old: {first_old_year}-{end_year}"
    )


def run_prediction(
    model,
    conditions,
    start_year,
    future_steps,
    temporal_evidence=None,
):
    """Run one evidence configuration and return a tidy time-series table."""
    try:
        model.net.set_slice_count(future_steps)
        model.net.clear_all_evidence()
        model_conditions = {
            MODEL_INPUT_NODE_IDS.get(node_id, node_id): state
            for node_id, state in conditions.items()
        }
        model.set_current_conditions(model_conditions)
        for node_id, schedule in (temporal_evidence or {}).items():
            model_node_id = MODEL_INPUT_NODE_IDS.get(node_id, node_id)
            for slice_index, state in schedule.items():
                model.net.set_temporal_evidence(
                    model_node_id,
                    slice_index,
                    state,
                )
        predictions = model.predict_corrosion()

        records = []
        for slice_index in range(future_steps):
            probabilities = predictions[f"t={slice_index}"]
            records.append(
                {
                    "Time slice": slice_index,
                    "Year": start_year + slice_index * TIME_STEP_YEARS,
                    "Low": probabilities["Low"],
                    "Moderate": probabilities["Moderate"],
                    "High": probabilities["High"],
                }
            )
        return pd.DataFrame(records)
    finally:
        model.net.clear_all_evidence()


def identify_tsw_periods(frame):
    """Find contiguous periods where Moderate is the dominant risk state."""
    dominant = frame["Moderate"].ge(frame[["Low", "High"]].max(axis=1))
    positions = [index for index, value in enumerate(dominant) if value]
    if not positions:
        return []

    runs = [[positions[0]]]
    for position in positions[1:]:
        if position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])

    return [
        {
            "start": int(frame.iloc[run[0]]["Year"]),
            "end": int(frame.iloc[run[-1]]["Year"]),
            "points": len(run),
        }
        for run in runs
    ]


def format_tsw(periods):
    if not periods:
        return "Not identified"
    labels = []
    for period in periods:
        if period["start"] == period["end"]:
            labels.append(str(period["start"]))
        else:
            labels.append(f"{period['start']}-{period['end']}")
    return "; ".join(labels)


def dominant_terminal_state(frame):
    terminal = frame.iloc[-1]
    return max(TARGET_STATES, key=lambda state: terminal[state])


def academic_layout(fig, x_title, height, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Times New Roman, serif", "size": 14, "color": "#111111"},
        margin={"l": 72, "r": 28, "t": 72, "b": 64},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.015,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 13},
            "entrywidth": 82,
            "entrywidthmode": "pixels",
        }
        if legend
        else None,
    )
    xaxis_style = dict(
        showline=True,
        linewidth=1.2,
        linecolor="#111111",
        mirror=True,
        ticks="outside",
        ticklen=5,
        tickcolor="#111111",
        gridcolor="#E3E6E8",
        gridwidth=0.7,
        zeroline=False,
    )
    if x_title:
        xaxis_style["title_text"] = x_title
    fig.update_xaxes(**xaxis_style)
    fig.update_yaxes(
        showline=True,
        linewidth=1.2,
        linecolor="#111111",
        mirror=True,
        ticks="outside",
        ticklen=5,
        tickcolor="#111111",
        gridcolor="#E3E6E8",
        gridwidth=0.7,
        zeroline=False,
    )
    return fig


def add_tsw_bands(fig, periods, row=None, col=None, annotate=True):
    for index, period in enumerate(periods):
        start = period["start"]
        end = period["end"]
        if start == end:
            start -= TIME_STEP_YEARS * 0.35
            end += TIME_STEP_YEARS * 0.35

        kwargs = {
            "x0": start,
            "x1": end,
            "fillcolor": "#F1D59B",
            "opacity": 0.33,
            "line_width": 0,
            "layer": "below",
        }
        if row is not None and col is not None:
            kwargs.update({"row": row, "col": col})
        if annotate and index == 0:
            kwargs.update(
                {
                    "annotation_text": "TSW",
                    "annotation_position": "top left",
                    "annotation_font": {"size": 12, "color": "#5B4520"},
                }
            )
        fig.add_vrect(**kwargs)


def baseline_probability_figure(frame, periods):
    fig = go.Figure()
    add_tsw_bands(fig, periods)
    for state in TARGET_STATES:
        style = STATE_STYLE[state]
        fig.add_trace(
            go.Scatter(
                x=frame["Year"],
                y=frame[state],
                mode="lines+markers",
                name=state,
                line={"color": style["color"], "width": 2.2},
                marker={
                    "color": "#FFFFFF",
                    "line": {"color": style["color"], "width": 1.8},
                    "size": 7,
                    "symbol": style["symbol"],
                },
                hovertemplate=(
                    "Year %{x}<br>" + state + ": %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(title={"text": "External corrosion depth probability evolution", "x": 0})
    academic_layout(fig, "Calendar year", 510)
    fig.update_yaxes(title_text="Posterior probability", range=[0, 1], dtick=0.2)
    fig.update_xaxes(tickmode="array", tickvals=frame["Year"])
    return fig


def scenario_time_slice_bar_figure(results, selected_node, baseline_state):
    column_count = 2
    row_count = (len(results) + column_count - 1) // column_count
    subplot_titles = [
        f"{state} (Baseline)" if state == baseline_state else state
        for state in results
    ]
    subplot_titles.extend([""] * (row_count * column_count - len(subplot_titles)))
    fig = make_subplots(
        rows=row_count,
        cols=column_count,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12 if row_count > 1 else 0.08,
        horizontal_spacing=0.09,
    )

    for scenario_index, (scenario_state, frame) in enumerate(results.items()):
        row = scenario_index // column_count + 1
        col = scenario_index % column_count + 1
        for risk_state in TARGET_STATES:
            style = STATE_STYLE[risk_state]
            fig.add_trace(
                go.Bar(
                    x=frame["Year"],
                    y=frame[risk_state],
                    name=risk_state,
                    legendgroup=risk_state,
                    showlegend=scenario_index == 0,
                    marker={
                        "color": style["color"],
                        "line": {"color": "#FFFFFF", "width": 0.6},
                    },
                    hovertemplate=(
                        f"{selected_node}={scenario_state}<br>"
                        f"{risk_state}<br>Year %{{x}}<br>"
                        "Probability: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            tickmode="array",
            tickvals=frame["Year"],
            tickangle=-45,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Probability" if col == 1 else None,
            range=[0, 1],
            dtick=0.2,
            row=row,
            col=col,
        )

    figure_height = 230 + 300 * row_count
    fig.update_layout(
        title={
            "text": (
                f"Time-slice risk distributions by {NODE_LABELS[selected_node]} "
                f"({selected_node}) state"
            ),
            "x": 0,
        },
        barmode="group",
        bargap=0.16,
        bargroupgap=0.03,
    )
    academic_layout(fig, None, figure_height)
    fig.update_layout(margin={"l": 72, "r": 28, "t": 90, "b": 88})
    fig.add_annotation(
        text="Calendar year",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"family": "Times New Roman, serif", "size": 14, "color": "#111111"},
    )
    return fig


def primary_tsw(periods):
    if not periods:
        return None
    return max(periods, key=lambda period: period["points"])


def scenario_tsw_bar_figure(results, selected_node, baseline_state):
    labels = []
    starts = []
    widths = []
    text = []
    colors = []
    first_year = min(int(frame.iloc[0]["Year"]) for frame in results.values())
    final_year = max(int(frame.iloc[-1]["Year"]) for frame in results.values())

    for scenario_state, frame in results.items():
        period = primary_tsw(identify_tsw_periods(frame))
        labels.append(
            f"{scenario_state} (Baseline)"
            if scenario_state == baseline_state
            else scenario_state
        )
        colors.append("#174A6E" if scenario_state == baseline_state else "#86A9BF")
        if period is None:
            starts.append(first_year)
            widths.append(0.01)
            text.append("Not identified")
        else:
            starts.append(period["start"])
            widths.append(max(period["end"] - period["start"], 0.35))
            text.append(
                str(period["start"])
                if period["start"] == period["end"]
                else f"{period['start']}-{period['end']}"
            )

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=widths,
            base=starts,
            orientation="h",
            marker={"color": colors, "line": {"color": "#FFFFFF", "width": 1}},
            text=text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>TSW: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": (
                f"Transitional Susceptibility Window by "
                f"{NODE_LABELS[selected_node]} ({selected_node}) state"
            ),
            "x": 0,
        },
        showlegend=False,
    )
    academic_layout(
        fig,
        "Calendar year",
        max(360, 210 + 48 * len(results)),
        legend=False,
    )
    fig.update_xaxes(
        range=[first_year - 1, final_year + 2],
        tickmode="linear",
        tick0=first_year,
        dtick=TIME_STEP_YEARS,
    )
    fig.update_yaxes(
        title_text="",
        categoryorder="array",
        categoryarray=labels[::-1],
    )
    return fig


def dbn_topology_figure():
    root = ET.parse(MODEL_PATH).getroot()
    genie = root.find("./extensions/genie")
    node_elements = list(root.find("nodes"))

    positions = {}
    display_names = {}
    if genie is not None:
        for node in genie.findall("node"):
            position = node.findtext("position")
            if not position:
                continue
            x1, y1, x2, y2 = [float(value) for value in position.split()]
            positions[node.get("id")] = ((x1 + x2) / 2, -(y1 + y2) / 2)
            display_names[node.get("id")] = node.findtext("name") or node.get("id")

    edge_x = []
    edge_y = []
    for node in node_elements:
        child_id = node.get("id")
        parents = (node.findtext("parents") or "").split()
        if child_id not in positions:
            continue
        child_x, child_y = positions[child_id]
        for parent_id in parents:
            if parent_id not in positions:
                continue
            parent_x, parent_y = positions[parent_id]
            edge_x.extend([parent_x, child_x, None])
            edge_y.extend([parent_y, child_y, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#9FB2BE", "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    internal_to_paper = {
        internal_id: paper_id
        for paper_id, internal_id in MODEL_INPUT_NODE_IDS.items()
    }
    categories = {
        "Input factors": {
            "ids": [], "color": "#2F7FA3", "symbol": "square", "size": 18,
        },
        "Risk aggregates": {
            "ids": [], "color": "#5F5A9B", "symbol": "diamond", "size": 21,
        },
        "Target E1": {
            "ids": [], "color": "#D2693C", "symbol": "star", "size": 27,
        },
    }
    for node_id in positions:
        if node_id == TARGET_NODE:
            categories["Target E1"]["ids"].append(node_id)
        elif node_id.startswith("X") and node_id[1:].isdigit():
            categories["Input factors"]["ids"].append(node_id)
        else:
            categories["Risk aggregates"]["ids"].append(node_id)

    for category, style in categories.items():
        ids = style["ids"]
        fig.add_trace(
            go.Scatter(
                x=[positions[node_id][0] for node_id in ids],
                y=[positions[node_id][1] for node_id in ids],
                mode="markers+text",
                name=category,
                text=[internal_to_paper.get(node_id, node_id) for node_id in ids],
                textposition="top center",
                textfont={"family": "Arial", "size": 10, "color": "#263844"},
                marker={
                    "color": style["color"],
                    "size": style["size"],
                    "symbol": style["symbol"],
                    "line": {"color": "#FFFFFF", "width": 1.2},
                },
                customdata=[display_names.get(node_id, node_id) for node_id in ids],
                hovertemplate="%{text}<br>%{customdata}<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": "Dynamic Bayesian network topology", "x": 0},
        height=680,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Arial", "color": "#1E2C36"},
        margin={"l": 20, "r": 20, "t": 75, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        hovermode="closest",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def probability_table(frame):
    display = frame.copy()
    display["Time slice"] = display["Time slice"].astype(int)
    display["Year"] = display["Year"].astype(int)
    for state in TARGET_STATES:
        display[state] = display[state].map(lambda value: f"{value:.4f}")
    return display


def describe_tsw_change(periods, baseline_periods):
    period = primary_tsw(periods)
    baseline = primary_tsw(baseline_periods)

    if baseline is None and period is None:
        return "No change: no TSW is identified within the prediction horizon."
    if baseline is None:
        return f"A TSW emerges at {format_tsw(periods)}; the baseline has no TSW."
    if period is None:
        return "The TSW is no longer identified within the prediction horizon."
    if period["start"] == baseline["start"] and period["end"] == baseline["end"]:
        return f"No change relative to the baseline TSW ({format_tsw(baseline_periods)})."

    start_shift = period["start"] - baseline["start"]
    end_shift = period["end"] - baseline["end"]
    duration_change = (
        period["end"] - period["start"]
        - (baseline["end"] - baseline["start"])
    )

    if start_shift == end_shift:
        direction = "delayed" if start_shift > 0 else "advanced"
        return (
            f"The TSW is {direction} by {abs(start_shift)} years "
            f"({format_tsw(periods)} versus {format_tsw(baseline_periods)})."
        )

    if start_shift > 0:
        start_text = f"onset delayed by {start_shift} years"
    elif start_shift < 0:
        start_text = f"onset advanced by {abs(start_shift)} years"
    else:
        start_text = "onset unchanged"

    if end_shift > 0:
        end_text = f"ending year delayed by {end_shift} years"
    elif end_shift < 0:
        end_text = f"ending year advanced by {abs(end_shift)} years"
    else:
        end_text = "ending year unchanged"

    if duration_change > 0:
        duration_text = f"window lengthened by {duration_change} years"
    elif duration_change < 0:
        duration_text = f"window shortened by {abs(duration_change)} years"
    else:
        duration_text = "window duration unchanged"

    return f"TSW {start_text}; {end_text}; {duration_text}."


def format_onset_shift(periods, baseline_periods):
    period = primary_tsw(periods)
    baseline = primary_tsw(baseline_periods)
    if period is None or baseline is None:
        return "N/A"
    shift = period["start"] - baseline["start"]
    if shift > 0:
        return f"+{shift} years (delayed)"
    if shift < 0:
        return f"{shift} years (advanced)"
    return "0 years (no change)"


def scenario_summary_table(results, baseline_state):
    baseline_periods = identify_tsw_periods(results[baseline_state])
    records = []
    for scenario_state, frame in results.items():
        periods = identify_tsw_periods(frame)
        terminal = frame.iloc[-1]
        records.append(
            {
                "Scenario state": scenario_state,
                "Role": "Baseline" if scenario_state == baseline_state else "Alternative",
                "TSW": format_tsw(periods),
                "TSW onset shift": (
                    "Reference"
                    if scenario_state == baseline_state
                    else format_onset_shift(periods, baseline_periods)
                ),
                "Terminal Low": f"{terminal['Low']:.4f}",
                "Terminal Moderate": f"{terminal['Moderate']:.4f}",
                "Terminal High": f"{terminal['High']:.4f}",
                "TSW interpretation": (
                    "Baseline reference"
                    if scenario_state == baseline_state
                    else describe_tsw_change(periods, baseline_periods)
                ),
            }
        )
    return pd.DataFrame(records)


def figure_config(filename):
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 1000,
            "width": 1600,
            "scale": 2,
        },
    }


def render_input_workspace(model, model_slices):
    with st.expander(
        "Pipeline Configuration",
        expanded=False,
        icon=":material/tune:",
    ):
        st.caption(
            "Set the assessment period and baseline evidence. These values are shared "
            "by every calculation in the current workspace."
        )
        period_col, age_col, slice_col = st.columns(3)
        with period_col:
            assessment_start_year = st.number_input(
                "Assessment start year",
                min_value=1900,
                max_value=2200,
                value=2007,
                step=1,
                help="Calendar year represented by time slice t=0.",
            )
        with age_col:
            current_pipeline_age = st.number_input(
                "Current pipeline age (years)",
                min_value=0,
                max_value=150,
                value=26,
                step=1,
                help="Used to determine the initial X11 state.",
            )
        with slice_col:
            future_steps = st.slider(
                "Number of time slices",
                min_value=2,
                max_value=model_slices,
                value=model_slices,
                step=1,
                help=f"Each time slice represents {TIME_STEP_YEARS} years.",
            )

        age_state = pipeline_age_state(current_pipeline_age)
        age_schedule = build_age_schedule(current_pipeline_age, future_steps)
        age_schedule_label = format_age_schedule(
            age_schedule,
            int(assessment_start_year),
        )
        st.info(f"Pipeline age schedule (X11): **{age_schedule_label}**")

        base_conditions = {"X11": age_state}
        st.markdown('<div class="section-heading">Baseline evidence</div>', unsafe_allow_html=True)
        group_rows = (NODE_GROUPS[:3], NODE_GROUPS[3:])
        for group_row in group_rows:
            columns = st.columns(len(group_row))
            for column, (group_name, node_ids) in zip(columns, group_row):
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{group_name}**")
                        for node_id in node_ids:
                            states = get_input_states(model, node_id)
                            default_state = DEFAULT_STATES.get(node_id, states[0])
                            default_index = (
                                states.index(default_state)
                                if default_state in states
                                else 0
                            )
                            base_conditions[node_id] = st.selectbox(
                                f"{NODE_LABELS[node_id]} ({node_id})",
                                states,
                                index=default_index,
                                key=f"baseline_{node_id}",
                            )

        st.caption(
            f"20 input nodes configured | Prediction horizon: "
            f"{assessment_start_year}-"
            f"{assessment_start_year + (future_steps - 1) * TIME_STEP_YEARS}"
        )
    return {
        "assessment_start_year": int(assessment_start_year),
        "current_pipeline_age": current_pipeline_age,
        "future_steps": future_steps,
        "age_schedule": age_schedule,
        "age_schedule_label": age_schedule_label,
        "base_conditions": base_conditions,
    }


def render_network_map():
    with st.expander(
        "DBN Network Map",
        expanded=False,
        icon=":material/account_tree:",
    ):
        st.caption(
            "Interactive topology reconstructed from the node positions and parent relationships "
            "stored in the original GeNIe XDSL file. Hover over a node for its model name."
        )
        st.plotly_chart(
            dbn_topology_figure(),
            width="stretch",
            config=figure_config("dbn_network_topology"),
        )


def render_baseline_workspace(
    baseline_frame,
    baseline_tsw,
    terminal_row,
    future_steps,
    age_schedule_label,
    base_conditions,
):
    with st.expander(
        "Risk Forecast",
        expanded=False,
        icon=":material/show_chart:",
    ):
        st.markdown(
            """
            <div class="method-note">
                X1-X10 and X12-X20 are entered at t=0. X11 follows the known age
                schedule. TSW is identified only when Moderate is at least as probable
                as both Low and High across consecutive time slices.
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_tsw, metric_terminal, metric_state, metric_horizon = st.columns(4)
        metric_tsw.metric("Transitional Susceptibility Window", format_tsw(baseline_tsw))
        metric_terminal.metric(
            f"P(High) in {int(terminal_row['Year'])}",
            f"{terminal_row['High']:.1%}",
        )
        metric_state.metric("Terminal dominant state", dominant_terminal_state(baseline_frame))
        metric_horizon.metric(
            "Prediction horizon",
            f"{(future_steps - 1) * TIME_STEP_YEARS} years",
        )
        st.plotly_chart(
            baseline_probability_figure(baseline_frame, baseline_tsw),
            width="stretch",
            config=figure_config("baseline_probability_evolution"),
        )
        st.markdown(
            "<p class='figure-caption'><b>Figure 1.</b> Posterior probability trajectories of the three external corrosion depth states. The shaded region denotes the TSW.</p>",
            unsafe_allow_html=True,
        )

    with st.expander(
        "Detailed Results & Export",
        expanded=False,
        icon=":material/download:",
    ):
        detail_col, input_col = st.columns([1.35, 1])
        with detail_col:
            st.markdown('<div class="section-heading">Numerical results</div>', unsafe_allow_html=True)
            st.dataframe(probability_table(baseline_frame), width="stretch", hide_index=True)
            st.download_button(
                "Download baseline results (CSV)",
                data=baseline_frame.to_csv(index=False).encode("utf-8"),
                file_name="baseline_time_evolution.csv",
                mime="text/csv",
            )
        with input_col:
            st.markdown('<div class="section-heading">Evidence specification</div>', unsafe_allow_html=True)
            evidence_table = pd.DataFrame(
                [
                    {
                        "Node": node_id,
                        "Input factor": NODE_LABELS[node_id],
                        "State": (
                            age_schedule_label
                            if node_id == "X11"
                            else base_conditions[node_id]
                        ),
                    }
                    for node_id in NODE_LABELS
                ]
            )
            st.dataframe(evidence_table, width="stretch", hide_index=True)

def render_scenario_workspace(
    model,
    baseline_frame,
    assessment_start_year,
    future_steps,
    age_schedule,
    base_conditions,
):
    with st.expander(
        "Factor Selection",
        expanded=False,
        icon=":material/alt_route:",
    ):
        st.caption(
            "Choose one factor. Its baseline state is compared with every alternative "
            "while the other 19 input nodes remain fixed."
        )
        control_col, status_col = st.columns([1.25, 1])
        with control_col:
            scenario_node = st.selectbox(
                "Node selected for comparison",
                list(NODE_LABELS),
                index=list(NODE_LABELS).index("X10"),
                format_func=lambda node_id: f"{node_id} - {NODE_LABELS[node_id]}",
                key="scenario_node",
            )
            available_scenario_states = get_input_states(model, scenario_node)
            baseline_scenario_state = base_conditions[scenario_node]
            scenario_states = [baseline_scenario_state] + [
                state
                for state in available_scenario_states
                if state != baseline_scenario_state
            ]
        with status_col:
            st.metric("Reference state", f"{scenario_node} = {baseline_scenario_state}")
            st.caption(
                f"19 nodes fixed | {len(scenario_states)} states compared"
            )

    scenario_results = {}
    for scenario_state in scenario_states:
        if scenario_state == baseline_scenario_state:
            scenario_results[scenario_state] = baseline_frame.copy()
        else:
            scenario_conditions = dict(base_conditions)
            scenario_conditions[scenario_node] = scenario_state
            scenario_results[scenario_state] = run_prediction(
                model,
                scenario_conditions,
                assessment_start_year,
                future_steps,
                temporal_evidence=(
                    None if scenario_node == "X11" else {"X11": age_schedule}
                ),
            )

    baseline_scenario_tsw = identify_tsw_periods(
        scenario_results[baseline_scenario_state]
    )

    with st.expander(
        "Comparative Risk Profiles",
        expanded=False,
        icon=":material/bar_chart:",
    ):
        st.info(
            f"Reference: **{scenario_node} = {baseline_scenario_state}** | "
            f"TSW: **{format_tsw(baseline_scenario_tsw)}**"
        )
        st.plotly_chart(
            scenario_time_slice_bar_figure(
                scenario_results,
                scenario_node,
                baseline_scenario_state,
            ),
            width="stretch",
            config=figure_config(f"scenario_{scenario_node}_time_slice_bars"),
        )
        st.markdown(
            f"<p class='figure-caption'><b>Figure 2.</b> Low, Moderate, and High probability bars for all {future_steps} selected time slices under each {NODE_LABELS[scenario_node]} scenario.</p>",
            unsafe_allow_html=True,
        )

    with st.expander(
        "TSW Decision Summary",
        expanded=False,
        icon=":material/timeline:",
    ):
        st.plotly_chart(
            scenario_tsw_bar_figure(
                scenario_results,
                scenario_node,
                baseline_scenario_state,
            ),
            width="stretch",
            config=figure_config(f"scenario_{scenario_node}_tsw_bars"),
        )
        st.markdown(
            "<p class='figure-caption'><b>Figure 3.</b> TSW intervals from direct Moderate-state dominance. The dark bar is the reference scenario.</p>",
            unsafe_allow_html=True,
        )
        for scenario_state, frame in scenario_results.items():
            if scenario_state == baseline_scenario_state:
                continue
            st.markdown(
                f"- **{scenario_node} = {scenario_state}:** "
                f"{describe_tsw_change(identify_tsw_periods(frame), baseline_scenario_tsw)}"
            )
        summary_table = scenario_summary_table(
            scenario_results,
            baseline_scenario_state,
        )
        st.dataframe(summary_table, width="stretch", hide_index=True)

    with st.expander(
        "Data Export",
        expanded=False,
        icon=":material/download:",
    ):
        st.caption(
            "Export either the complete probability record for every selected time slice "
            "or the concise TSW comparison summary."
        )
        long_results = pd.concat(
            [
                frame.assign(
                    **{
                        "Selected node": scenario_node,
                        "Scenario state": scenario_state,
                        "Scenario role": (
                            "Baseline"
                            if scenario_state == baseline_scenario_state
                            else "Alternative"
                        ),
                    }
                )
                for scenario_state, frame in scenario_results.items()
            ],
            ignore_index=True,
        )
        probability_export, summary_export = st.columns(2)
        with probability_export:
            st.markdown("**All time-slice probabilities**")
            st.caption(
                "Low, Moderate, and High probabilities for every scenario and year."
            )
            st.download_button(
                "Download probability results (CSV)",
                data=long_results.to_csv(index=False).encode("utf-8"),
                file_name=f"scenario_probability_results_{scenario_node}.csv",
                mime="text/csv",
            )
        with summary_export:
            st.markdown("**TSW comparison summary**")
            st.caption(
                "Scenario roles, TSW intervals, terminal probabilities, and dominant states."
            )
            st.download_button(
                "Download TSW summary (CSV)",
                data=summary_table.to_csv(index=False).encode("utf-8"),
                file_name=f"scenario_tsw_summary_{scenario_node}.csv",
                mime="text/csv",
            )


def dashboard_home_html(model_nodes, model_slices):
    return f"""
        <div class="dashboard-intro">
            <div class="dashboard-eyebrow">Assessment workspace</div>
            <div class="dashboard-title">Choose an analysis module</div>
            <div class="dashboard-lead">
                Enter a focused workspace for dynamic risk prediction or controlled scenario
                comparison. Both workspaces use the same validated DBN baseline.
            </div>
        </div>
        <div class="dashboard-grid">
            <a class="dashboard-card baseline" href="?view=baseline" target="_self">
                <div class="dashboard-card-top">
                    <div class="dashboard-card-icon">{icon_svg("analysis")}</div>
                </div>
                <div class="dashboard-card-title">Baseline Evolution</div>
                <div class="dashboard-card-text">
                    Configure X1-X20 evidence and predict Low, Moderate, and High external
                    corrosion depth probabilities across the selected time horizon.
                </div>
                <div class="dashboard-card-pills">
                    <span class="dashboard-pill">20 inputs</span>
                    <span class="dashboard-pill">{model_slices} slices</span>
                    <span class="dashboard-pill">TSW detection</span>
                </div>
                <div class="dashboard-card-action">Open workspace &rarr;</div>
            </a>
            <a class="dashboard-card scenario" href="?view=scenario" target="_self">
                <div class="dashboard-card-top">
                    <div class="dashboard-card-icon">{icon_svg("scenario")}</div>
                </div>
                <div class="dashboard-card-title">Scenario Explorer</div>
                <div class="dashboard-card-text">
                    Hold 19 nodes fixed, vary one selected factor, compare all time-slice
                    probability bars, and quantify TSW advancement or delay.
                </div>
                <div class="dashboard-card-pills">
                    <span class="dashboard-pill">One-factor analysis</span>
                    <span class="dashboard-pill">Baseline comparison</span>
                    <span class="dashboard-pill">TSW shift</span>
                </div>
                <div class="dashboard-card-action">Open workspace &rarr;</div>
            </a>
        </div>
    """


def back_to_dashboard_html():
    return '<a class="back-link" href="?" target="_self">&#8592;&nbsp; Back to dashboard</a>'


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Web implementation: 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    model = load_model()
    validate_model_contract(model)
except Exception as exc:
    st.error(f"The DBN model could not be loaded or validated: {exc}")
    st.stop()

model_slices = st.session_state.pipeline_dbn_max_slices
model_nodes = model.net.get_node_count()
requested_view = st.query_params.get("view", "home")
active_view = requested_view if requested_view in {"baseline", "scenario"} else "home"
header_class = "paper-header" if active_view == "home" else "paper-header compact"

st.markdown(
    f"""
    <div class="{header_class}">
        <div class="paper-heading-row">
            <div class="paper-heading-copy">
                <div class="paper-kicker">Web-enabled UPEC assessment platform | 2026</div>
                <div class="paper-title">A web-enabled dynamic Bayesian network probabilistic assessment of underground pipeline external corrosion</div>
                <div class="paper-subtitle">
                    Interactive dynamic risk prediction, Transitional Susceptibility Window identification,
                    and one-factor-at-a-time scenario comparison.
                </div>
                <div class="paper-authors">Gang Zhang &nbsp;|&nbsp; Nii Attoh-Okine</div>
                <div class="paper-affiliation">Department of Civil and Environmental Engineering, University of Maryland, College Park, Maryland, USA</div>
            </div>
            <div class="paper-network-mark">{pipeline_hero_svg()}</div>
        </div>
        <div class="metadata-row">
            <span class="metadata-item">{icon_svg("output", "metadata-icon")}Web implementation: 2026</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if active_view == "home":
    st.markdown(
        dashboard_home_html(model_nodes, model_slices),
        unsafe_allow_html=True,
    )
    render_footer()
    st.stop()

st.markdown(back_to_dashboard_html(), unsafe_allow_html=True)

if active_view == "baseline":
    st.markdown(
        module_header_html(
            "BASELINE WORKSPACE",
            "Dynamic Risk Workspace",
            "Configure the pipeline evidence, inspect the forecast, and export the results.",
            "analysis",
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        module_header_html(
            "SCENARIO WORKSPACE",
            "Counterfactual Comparison Workspace",
            "Hold the reference case fixed and examine one factor across all selected time slices.",
            "scenario",
        ),
        unsafe_allow_html=True,
    )

input_config = render_input_workspace(model, model_slices)
assessment_start_year = input_config["assessment_start_year"]
future_steps = input_config["future_steps"]
age_schedule = input_config["age_schedule"]
age_schedule_label = input_config["age_schedule_label"]
base_conditions = input_config["base_conditions"]

try:
    baseline_frame = run_prediction(
        model,
        base_conditions,
        assessment_start_year,
        future_steps,
        temporal_evidence={"X11": age_schedule},
    )
except Exception as exc:
    st.error(f"Baseline inference failed: {exc}")
    st.stop()

baseline_tsw = identify_tsw_periods(baseline_frame)
terminal_row = baseline_frame.iloc[-1]

if active_view == "baseline":
    render_baseline_workspace(
        baseline_frame,
        baseline_tsw,
        terminal_row,
        future_steps,
        age_schedule_label,
        base_conditions,
    )
    render_footer()
    st.stop()

    st.markdown(
        module_header_html(
            "BASELINE WORKSPACE",
            "Baseline Evolution Analysis",
            "Baseline DBN inference, risk trajectories, and TSW identification.",
            "analysis",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="method-note">
            X1-X10 and X12-X20 are treated as evidence at the initial time slice.
            X11 is entered as known temporal evidence at every time slice from the
            current pipeline age and the two-year interval.
            The TSW is identified wherever <b>P(Moderate) is greater than or equal to both
            P(Low) and P(High)</b> across consecutive time slices.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_tsw, metric_terminal, metric_state, metric_horizon = st.columns(4)
    metric_tsw.metric("Transitional Susceptibility Window", format_tsw(baseline_tsw))
    metric_terminal.metric(
        f"P(High) in {int(terminal_row['Year'])}",
        f"{terminal_row['High']:.1%}",
    )
    metric_state.metric("Terminal dominant state", dominant_terminal_state(baseline_frame))
    metric_horizon.metric(
        "Prediction horizon",
        f"{(future_steps - 1) * TIME_STEP_YEARS} years",
    )

    baseline_fig = baseline_probability_figure(baseline_frame, baseline_tsw)
    st.plotly_chart(
        baseline_fig,
        width="stretch",
        config=figure_config("baseline_probability_evolution"),
    )
    st.markdown(
        "<p class='figure-caption'><b>Figure 1.</b> Posterior probability trajectories of the three external corrosion depth states. The shaded region denotes the TSW.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        module_header_html(
            "MODULE 04",
            "Output Module",
            "Numerical probabilities, evidence specification, and downloadable results.",
            "output",
        ),
        unsafe_allow_html=True,
    )
    detail_col, input_col = st.columns([1.35, 1])
    with detail_col:
        st.markdown('<div class="section-heading">Numerical results</div>', unsafe_allow_html=True)
        st.dataframe(probability_table(baseline_frame), width="stretch", hide_index=True)
        st.download_button(
            "Download baseline results (CSV)",
            data=baseline_frame.to_csv(index=False).encode("utf-8"),
            file_name="baseline_time_evolution.csv",
            mime="text/csv",
        )

    with input_col:
        st.markdown('<div class="section-heading">Evidence specification</div>', unsafe_allow_html=True)
        evidence_table = pd.DataFrame(
            [
                {
                    "Node": node_id,
                    "Input factor": NODE_LABELS[node_id],
                    "State": (
                        age_schedule_label
                        if node_id == "X11"
                        else base_conditions[node_id]
                    ),
                }
                for node_id in NODE_LABELS
            ]
        )
        st.dataframe(evidence_table, width="stretch", hide_index=True)


if active_view == "scenario":
    try:
        render_scenario_workspace(
            model,
            baseline_frame,
            assessment_start_year,
            future_steps,
            age_schedule,
            base_conditions,
        )
    except Exception as exc:
        st.error(f"Scenario inference failed: {exc}")
        st.stop()
    render_footer()
    st.stop()

    st.markdown(
        module_header_html(
            "SCENARIO WORKSPACE",
            "Scenario Explorer",
            "Select one factor while all remaining baseline evidence stays fixed.",
            "scenario",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="method-note">
            Select one node from X1-X20. Every non-selected node remains fixed at the
            baseline value shown in the left panel. Only the selected node is varied,
            so differences between bars can be attributed to that factor within the DBN model.
            The baseline is always the current state selected in the left panel.
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col, status_col = st.columns([1.25, 1])
    with control_col:
        scenario_node = st.selectbox(
            "Node selected for comparison",
            list(NODE_LABELS),
            index=list(NODE_LABELS).index("X10"),
            format_func=lambda node_id: f"{node_id} - {NODE_LABELS[node_id]}",
            key="scenario_node",
        )
        available_scenario_states = get_input_states(model, scenario_node)
        baseline_scenario_state = base_conditions[scenario_node]
        scenario_states = [baseline_scenario_state] + [
            state
            for state in available_scenario_states
            if state != baseline_scenario_state
        ]

    with status_col:
        fixed_count = len(NODE_LABELS) - 1
        st.metric(
            "Baseline scenario",
            f"{scenario_node} = {baseline_scenario_state}",
        )
        st.caption(
            f"**{fixed_count}** other input nodes remain fixed.  "
            f"\nAll **{len(scenario_states)}** states of {scenario_node} are compared."
        )
        if scenario_node == "X11":
            st.caption(
                "For X11 scenarios, the selected state is applied at t=0 and "
                "subsequent states follow the DBN transition model."
            )

    scenario_results = {}
    try:
        for scenario_state in scenario_states:
            if scenario_state == baseline_scenario_state:
                scenario_results[scenario_state] = baseline_frame.copy()
            else:
                scenario_conditions = dict(base_conditions)
                scenario_conditions[scenario_node] = scenario_state
                scenario_results[scenario_state] = run_prediction(
                    model,
                    scenario_conditions,
                    int(assessment_start_year),
                    future_steps,
                    temporal_evidence=(
                        None
                        if scenario_node == "X11"
                        else {"X11": age_schedule}
                    ),
                )
    except Exception as exc:
        st.error(f"Scenario inference failed: {exc}")
        st.stop()

    st.markdown(
        module_header_html(
            "MODULE 04",
            "Comparative Analysis Module",
            "All-time-slice probability bars and scenario-specific TSW intervals.",
            "analysis",
        ),
        unsafe_allow_html=True,
    )
    baseline_scenario_tsw = identify_tsw_periods(
        scenario_results[baseline_scenario_state]
    )
    st.info(
        f"Baseline: **{scenario_node} = {baseline_scenario_state}**; "
        f"TSW = **{format_tsw(baseline_scenario_tsw)}**. "
        f"All other X1-X20 inputs remain at their baseline values."
    )

    time_slice_bar_fig = scenario_time_slice_bar_figure(
        scenario_results,
        scenario_node,
        baseline_scenario_state,
    )
    st.plotly_chart(
        time_slice_bar_fig,
        width="stretch",
        config=figure_config(f"scenario_{scenario_node}_time_slice_bars"),
    )
    st.markdown(
        f"<p class='figure-caption'><b>Figure 2.</b> Low, Moderate, and High probability bars for all {future_steps} selected time slices under each {NODE_LABELS[scenario_node]} ({scenario_node}) scenario. The current input state is marked as Baseline.</p>",
        unsafe_allow_html=True,
    )

    tsw_bar_fig = scenario_tsw_bar_figure(
        scenario_results,
        scenario_node,
        baseline_scenario_state,
    )
    st.plotly_chart(
        tsw_bar_fig,
        width="stretch",
        config=figure_config(f"scenario_{scenario_node}_tsw_bars"),
    )
    st.markdown(
        "<p class='figure-caption'><b>Figure 3.</b> TSW intervals identified directly from Moderate-state dominance. The dark bar is the Baseline.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        module_header_html(
            "MODULE 05",
            "Output & Interpretation Module",
            "Quantified TSW changes, scenario summary, and downloadable data.",
            "output",
        ),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-heading">TSW change interpretation</div>', unsafe_allow_html=True)
    for scenario_state, frame in scenario_results.items():
        if scenario_state == baseline_scenario_state:
            continue
        scenario_periods = identify_tsw_periods(frame)
        st.markdown(
            f"- **{scenario_node} = {scenario_state}:** "
            f"{describe_tsw_change(scenario_periods, baseline_scenario_tsw)}"
        )

    st.markdown('<div class="section-heading">Scenario comparison summary</div>', unsafe_allow_html=True)
    summary_table = scenario_summary_table(
        scenario_results,
        baseline_scenario_state,
    )
    st.dataframe(summary_table, width="stretch", hide_index=True)

    long_results = pd.concat(
        [
            frame.assign(
                **{
                    "Selected node": scenario_node,
                    "Scenario state": scenario_state,
                    "Scenario role": (
                        "Baseline"
                        if scenario_state == baseline_scenario_state
                        else "Alternative"
                    ),
                }
            )
            for scenario_state, frame in scenario_results.items()
        ],
        ignore_index=True,
    )
    st.download_button(
        "Download scenario results (CSV)",
        data=long_results.to_csv(index=False).encode("utf-8"),
        file_name=f"scenario_analysis_{scenario_node}.csv",
        mime="text/csv",
    )


render_footer()
