"""Dark glassmorphism theme + reusable UI components for the Streamlit app.

Visual language borrowed from reference_content/Credit_Risk_Pipeline_Teaching_Demo.html
(radial-gradient background, glass-panel cards, pill badges, gradient bar meters),
adapted to Streamlit via CSS injection + st.container(key=...) targeting.
"""

import streamlit as st

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }

:root {
  --panel: rgba(255,255,255,0.055);
  --line: rgba(255,255,255,0.14);
  --muted: #aab3c5;
  --accent: #38bdf8;
  --accent2: #a78bfa;
  --good: #22c55e;
  --warn: #f59e0b;
  --bad: #ef4444;
}

.stApp {
  background:
    radial-gradient(circle at 15% 0%, rgba(56,189,248,0.16), transparent 32%),
    radial-gradient(circle at 85% 8%, rgba(167,139,250,0.14), transparent 34%),
    linear-gradient(135deg, #0f172a, #111827 55%, #0b1120);
}

section[data-testid="stSidebar"] {
  background: rgba(2,6,23,0.80);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--muted); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #e5e7eb; }

/* glass card containers, targeted by st.container(key="card-...") */
div[class*="st-key-card"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 20px 24px 8px;
  box-shadow: 0 18px 40px rgba(0,0,0,.18);
  margin-bottom: 8px;
}

div[class*="st-key-hero"] {
  background:
    radial-gradient(circle at 90% -10%, rgba(56,189,248,0.14), transparent 40%),
    linear-gradient(135deg, rgba(56,189,248,.08), rgba(167,139,250,.08));
  border: 1px solid var(--line);
  border-radius: 26px;
  padding: 30px 34px 18px;
  margin-bottom: 20px;
}

/* pill badges */
.eyebrow {
  display:inline-flex; padding:6px 12px; border-radius:999px;
  font-size:12.5px; letter-spacing:.02em; margin-bottom:10px;
  background: rgba(255,255,255,0.06); color: var(--muted); border:1px solid var(--line);
}
.pill {
  display:inline-flex; padding:5px 12px; border-radius:999px;
  font-size:12.5px; font-weight:600; margin-right:6px;
  background: rgba(56,189,248,0.14); color:#bae6fd; border:1px solid rgba(56,189,248,.28);
}
.pill.green  { background: rgba(34,197,94,.14);  color:#bbf7d0; border-color: rgba(34,197,94,.30); }
.pill.orange { background: rgba(245,158,11,.14); color:#fde68a; border-color: rgba(245,158,11,.30); }
.pill.purple { background: rgba(167,139,250,.14); color:#ddd6fe; border-color: rgba(167,139,250,.30); }
.pill.muted  { background: rgba(255,255,255,.06); color: var(--muted); border-color: var(--line); }

/* metric tiles */
.tile-value { font-size: 27px; font-weight:800; color: #ffffff; line-height:1.15; }
.tile-label { font-size:13px; color: var(--muted); margin-top:2px; }

/* gradient bar meters, used for distributions */
.bar-row { display:grid; grid-template-columns: 150px 1fr 46px; gap:10px; align-items:center; font-size:13.5px; color:var(--muted); margin:7px 0; }
.bar-bg { height:11px; border-radius:999px; background: rgba(255,255,255,.10); overflow:hidden; }
.bar-fill { height:100%; border-radius:999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.bar-fill.green  { background: linear-gradient(90deg, #16a34a, #22c55e); }
.bar-fill.orange { background: linear-gradient(90deg, #d97706, #f59e0b); }

/* section heading inside cards */
.card-title { font-size: 18px; font-weight:700; color:#e5e7eb; margin-bottom:4px; }
.card-sub { color: var(--muted); font-size: 13.5px; margin-bottom: 10px; }

.note {
  border-left: 4px solid var(--accent);
  padding: 12px 14px;
  background: rgba(56,189,248,0.07);
  border-radius: 12px;
  color: var(--muted);
  font-size: 14px;
  margin: 10px 0;
}
.note.warning { border-left-color: var(--warn); background: rgba(245,158,11,0.08); }

/* Streamlit tab bar restyle */
button[data-baseweb="tab"] { color: var(--muted) !important; font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color: #e5e7eb !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
</style>
"""


def inject_theme() -> None:
    st.html(CSS)


def eyebrow(text: str) -> None:
    st.markdown(f'<span class="eyebrow">{text}</span>', unsafe_allow_html=True)


def pill(text: str, variant: str = "default") -> str:
    cls = "pill" + (f" {variant}" if variant != "default" else "")
    return f'<span class="{cls}">{text}</span>'


def tile(value, label: str) -> None:
    st.markdown(f'<div class="tile-value">{value}</div><div class="tile-label">{label}</div>', unsafe_allow_html=True)


def bar_row(label: str, value: float, max_value: float, color_class: str = "", value_fmt: str = "{:.0f}") -> None:
    pct = 0 if max_value == 0 else round(100 * value / max_value)
    st.markdown(
        f'<div class="bar-row"><span>{label}</span>'
        f'<div class="bar-bg"><div class="bar-fill {color_class}" style="width:{pct}%"></div></div>'
        f'<span>{value_fmt.format(value)}</span></div>',
        unsafe_allow_html=True,
    )


def card_title(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="card-sub">{subtitle}</div>', unsafe_allow_html=True)


def note(text: str, variant: str = "default") -> None:
    cls = "note" + (f" {variant}" if variant != "default" else "")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
