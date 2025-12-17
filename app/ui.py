import base64
import html
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root (fake-news-detector-distilbert)
sys.path.insert(0, str(ROOT_DIR))

from inference import predict
from db import init_db, save_prediction, fetch_filtered, fetch_sources, fetch_stats, delete_prediction

#-----------------------#
# Page config + DB init #
#-----------------------#
st.set_page_config(page_title="Fake News Detector", layout="wide")
init_db()

#----------------#
#    Helpers     #
#----------------#
def _find_logo_path() -> Path | None:
    logo_path = ROOT_DIR / "app" / "assets" / "upm-logo.png"
    return logo_path if logo_path.exists() else None


def _img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def category_class(category: str) -> str:
    cat = (category or "").lower()
    if "false" in cat:
        return "cat-false"
    if "ambiguous" in cat:
        return "cat-amb"
    if "true" in cat:
        return "cat-true"
    return "cat-amb"


def ensure_state_defaults():
    st.session_state.setdefault("form_version", 0)
    st.session_state.setdefault("last_result", None)


def reset_form():
    st.session_state["form_version"] += 1
    st.session_state["last_result"] = None
    st.rerun()


def render_rows(rows, context: str = "all"):
    if not rows:
        st.info("No predictions found for current filters.")
        return

    for r in rows:
        label = (r.get("label") or "").strip()
        conf = float(r.get("confidence_pct") or 0.0)
        created_at = (r.get("created_at") or "").strip()
        rid = int(r.get("id"))

        source = (r.get("source") or "").strip()
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        category = (r.get("category") or "").strip()

        full_text = body if body else (r.get("input_text") or "").strip()
        full_text_safe = html.escape(full_text)

        badge_cls = category_class(category)

        h1, h2 = st.columns([6, 1])
        with h1:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-top">
                    <div class="meta">
                      <span class="rid">#{rid}</span>
                      <span class="dot">•</span>
                      <span class="time">{html.escape(created_at)}</span>
                      <span class="dot">•</span>
                      <span class="label">{html.escape(label)} ({conf:.2f}%)</span>
                    </div>
                    <div class="badge {badge_cls}">{html.escape(category)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with h2:
            st.write("")
            if st.button("Delete", key=f"del_{context}_{rid}", type="secondary"):
                delete_prediction(rid)
                st.success(f"Deleted #{rid}")
                st.rerun()

        # --- body: only Source + Title, and full body inside expander
        if source:
            st.caption(f"Source: {source}")
        if title:
            st.markdown(f"**Title:** {title}")

        with st.expander("Show full body"):
            st.markdown(full_text_safe)

        st.write("")  # spacing


# ---------------------------#
# Sticky Navbar + Global CSS #
# ---------------------------#
ensure_state_defaults()

logo_path = _find_logo_path()
logo_b64 = _img_to_base64(logo_path) if logo_path else ""

# Sidebar width used to offset navbar (approx; Streamlit varies by theme/viewport)
SIDEBAR_W = "21rem"
NAV_H = "5.4rem"

st.markdown(
    f"""
    <style>
      /* Hide default Streamlit header/footer */
      header[data-testid="stHeader"] {{
        display: none;
      }}
      footer {{
        display: none;
      }}

      :root {{
        --sidebar-w: {SIDEBAR_W};
        --nav-h: {NAV_H};
      }}

      /* Make room for custom fixed navbar */
      .block-container {{
        padding-top: calc(var(--nav-h) + 0.8rem);
      }}

      /* Navbar (default assumes sidebar expanded) */
      .upm-navbar {{
          position: fixed;
          top: 0;
          left: var(--sidebar-w);
          right: 0;
          height: var(--nav-h);
          z-index: 9999;
          backdrop-filter: blur(10px);
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 1.5rem;
          border-bottom: 2px solid;
        }}
      
      /* Dark theme */
        @media (prefers-color-scheme: dark) {{
          .upm-navbar {{
            background: rgba(15, 18, 24, 0.92);
            border-color: rgba(255, 255, 255, 0.08);
            color: white;
          }}
          .upm-right {{
            border-color: rgba(255, 255, 255, 0.10);
            color: white;
          }}
        }}
        
        /* Light theme */
        @media (prefers-color-scheme: light) {{
          .upm-navbar {{
            background: rgba(255, 255, 255, 0.92);
            border-color: rgba(0, 0, 0, 0.08);
            color: black;
          }}
          .upm-right {{
            border-color: rgba(0, 0, 0, 0.10);
            color: black;
          }}

      /* On small screens, sidebar collapses -> navbar should use full width */
      @media (max-width: 992px) {{
        .upm-navbar {{
          left: 0 !important;
        }}
      }}
      
      .upm-left {{
        display: flex;
        align-items: center;
        gap: 0.9rem;
        min-width: 300px;
      }}
      .upm-logo {{
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: rgba(255,255,255,0.06);
        display: grid;
        place-items: center;
        overflow: hidden;
      }}
      .upm-logo img {{
        width: 38px;
        height: 38px;
        object-fit: contain;
      }}
      .upm-title {{
        display: flex;
        flex-direction: column;
        line-height: 1.05;
      }}
      .upm-title h1 {{
        font-size: 1.15rem;
        margin: 0;
        font-weight: 800;
      }}
      .upm-title span {{
        font-size: 0.92rem;
        opacity: 0.75;
      }}
      .upm-right {{
        font-size: 0.95rem;
        opacity: 0.85;
        padding: 0.35rem 0.7rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 999px;
        white-space: nowrap;
      }}

      /* Prediction result box */
      .result-box {{
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 14px;
        margin-top: 14px;
        background: rgba(255,255,255,0.03);
      }}
      .result-title {{
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 6px;
      }}

      /* Cards for saved predictions */
      .card {{
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        margin-bottom: 14px;
        background: rgba(255,255,255,0.03);
      }}
      .card-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }}
      .meta {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 0.98rem;
      }}
      .rid {{
        font-weight: 800;
      }}
      .dot {{
        opacity: 0.55;
      }}
      .label {{
        font-weight: 650;
      }}

      .badge {{
        font-weight: 900;
        font-size: 1.05rem;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        white-space: nowrap;
      }}
      .cat-true {{
        color: #32d583;
        background: rgba(50,213,131,0.10);
        border-color: rgba(50,213,131,0.22);
      }}
      .cat-false {{
        color: #ff4d4f;
        background: rgba(255,77,79,0.12);
        border-color: rgba(255,77,79,0.25);
      }}
      .cat-amb {{
        color: #fbbf24;
        background: rgba(251,191,36,0.12);
        border-color: rgba(251,191,36,0.25);
      }}

      /* Make expander body text preserve newlines */
      .fulltext {{
        white-space: pre-wrap;
        line-height: 1.5;
        opacity: 0.92;
      }}
    </style>

    <div class="upm-navbar">
      <div class="upm-left">
        <div class="upm-logo">
          {"<img src='data:image/png;base64," + logo_b64 + "' />" if logo_b64 else ""}
        </div>
        <div class="upm-title">
          <h1>Fake News Detector — DistilBERT</h1>
          <span>Deep Learning and Software Engineering</span>
        </div>
      </div>
      <div class="upm-right">Universidad Politécnica de Madrid</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------#
# Sidebar filters #
# ----------------#
st.sidebar.header("Filters")

label_filter = st.sidebar.selectbox("Label", ["All", "Real", "Fake"], index=0)
sources = ["All"] + fetch_sources()
source_filter = st.sidebar.selectbox("Source", sources, index=0)
min_conf = st.sidebar.slider("Min confidence (%)", 0, 100, 0, 5)
date_from = st.sidebar.text_input("Date from (YYYY-MM-DD)", value="")
date_to = st.sidebar.text_input("Date to (YYYY-MM-DD)", value="")
limit = st.sidebar.selectbox("Max rows", [10, 20, 50, 100], index=1)

# ------------#
# Main layout #
# ------------#
left, right = st.columns([1.2, 1.0], gap="large")

#-------------#
#    Form     #
#-------------#
with left:
    st.subheader("Predict a News Article")

    fv = st.session_state["form_version"]

    with st.form(f"predict_form_{fv}", clear_on_submit=False):
        title = st.text_input(
            "News Title",
            placeholder="Enter the headline/title...",
            key=f"news_title_{fv}",
        )
        body = st.text_area(
            "News Body",
            height=220,
            placeholder="Paste the article body here...",
            key=f"news_body_{fv}",
        )
        source = st.text_input(
            "News Source (optional)",
            placeholder="Reuters, CNN, BBC...",
            key=f"news_source_{fv}",
        )
        date_str = st.text_input(
            "Date (optional)",
            placeholder="YYYY-MM-DD",
            key=f"news_date_{fv}",
        )

        # Buttons Predict and Clear.
        b1, b2, _spacer = st.columns([1, 1, 3])
        with b1:
            do_predict = st.form_submit_button("Predict")
        with b2:
            do_clear = st.form_submit_button("Clear")

    if do_clear:
        reset_form()

    if do_predict:
        if not title.strip() and not body.strip():
            st.warning("Please provide at least a Title or Body.")
        else:
            out = predict(title=title, body=body, source=source, date_str=date_str)
            st.session_state["last_result"] = out
            save_prediction(out)

    # Render last result (persist until next clear)
    if st.session_state.get("last_result"):
        out = st.session_state["last_result"]
        badge_cls = category_class(out.get("category", ""))

        st.markdown(
            f"""
            <div class="result-box">
              <div class="result-title">
                {"✅ Prediction: REAL" if out["label"] == "Real" else "❌ Prediction: FAKE"}
              </div>
              <div class="badge {badge_cls}" style="margin-top:10px;">{html.escape(out["category"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{out['confidence_pct']:.2f}%")
        c2.metric("Prob Real", f"{out['prob_real'] * 100:.2f}%")
        c3.metric("Prob Fake", f"{out['prob_fake'] * 100:.2f}%")

        with st.expander("Raw probabilities"):
            st.json({"prob_fake": out["prob_fake"], "prob_real": out["prob_real"]})

#------------#
# Overview   #
#------------#
with right:
    st.subheader("Overview")
    stats = fetch_stats()
    pct_real = (stats["real"] / stats["total"] * 100.0) if stats["total"] else 0.0

    a, b, c = st.columns(3)
    a.metric("Total Articles", stats["total"])
    b.metric("Real News", stats["real"])
    c.metric("Fake News", stats["fake"])

    st.metric("% of Real Articles", f"{pct_real:.1f}%")
    st.metric("Avg Confidence of Model", f"{stats['avg_conf']:.2f}%")
    st.info("Tip: Use sidebar filters to explore saved predictions.")

st.divider()
st.subheader("Saved Predictions")

tab_all, tab_real, tab_fake = st.tabs(["All", "Real", "Fake"])

#-------------------#
# Saved Predictions #
#-------------------#
def _fetch_rows(effective_label: str):
    return fetch_filtered(
        label=effective_label,
        source=source_filter,
        min_conf=min_conf,
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
        limit=limit,
    )


with tab_all:
    rows = _fetch_rows(label_filter)
    render_rows(rows, context="tab_all")

with tab_real:
    effective = "Real" if label_filter == "All" else label_filter
    rows = _fetch_rows(effective)
    render_rows(rows, context="tab_real")

with tab_fake:
    effective = "Fake" if label_filter == "All" else label_filter
    rows = _fetch_rows(effective)
    render_rows(rows, context="tab_fake")
