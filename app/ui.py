# app/ui.py
from pathlib import Path
import streamlit as st

from inference import predict
from db import init_db, save_prediction, fetch_filtered, fetch_sources, fetch_stats


# -------------------------
# Page config + DB init
# -------------------------
st.set_page_config(page_title="Fake News Detector · DistilBERT", layout="wide")
init_db()

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "upm-logo.png"


# -------------------------
# Styles
# -------------------------
st.markdown(
    """
    <style>
      /* Make main content a bit narrower for nicer reading */
      .block-container {
        padding-top: 3rem;
        max-width: 1200px;
      }

      /* Navbar container */
      .nav-wrap {
        width: 100%;
        margin-bottom: 1.2rem;
      }

      .nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.9rem 1.1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
      }

      .nav-left {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        min-width: 0;
      }

      .nav-title {
        font-size: 1.25rem;
        font-weight: 750;
        line-height: 1.15;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nav-subtitle {
        font-size: 0.92rem;
        opacity: 0.72;
        margin-top: 0.15rem;
      }

      .badge {
        font-size: 0.88rem;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.05);
        opacity: 0.92;
        white-space: nowrap;
      }

      /* Result card */
      .result-card {
        padding: 0.95rem 1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        margin-top: 0.75rem;
      }
      .result-title {
        font-size: 1.1rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
      }
      .muted {
        opacity: 0.78;
        font-size: 0.95rem;
      }

      /* Sidebar title spacing */
      section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_navbar():
    # Use columns for a stable alignment (logo | title | badge)
    c1, c2, c3 = st.columns([0.10, 0.65, 0.25], vertical_alignment="center")

    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=56)
        else:
            st.write("UPM")

    with c2:
        st.markdown(
            """
            <div>
              <div class="nav-title">Fake News Detector · DistilBERT</div>
              <div class="nav-subtitle">Deep Learning and Software Engineering</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """<div style="text-align:right;"><span class="badge">Universidad Politécnica de Madrid</span></div>""",
            unsafe_allow_html=True,
        )


def render_rows(rows):
    if not rows:
        st.info("No predictions found for current filters.")
        return

    for r in rows:
        st.markdown(
            f"**#{r['id']}** · {r['created_at']} · "
            f"**{r['label']}** ({r['confidence_pct']:.2f}%) · `{r['category']}`"
        )
        if r.get("source"):
            st.caption(f"Source: {r['source']}")

        if r.get("title"):
            st.write("**Title:**", r["title"])

        preview = r.get("input_text", "") or ""
        st.caption((preview[:200] + "…") if len(preview) > 200 else preview)

        with st.expander("Show full input"):
            st.write(preview)

        st.write("---")


def label_badge(label: str) -> str:
    label = (label or "").strip().lower()
    return "✅ REAL" if label == "real" else "❌ FAKE"


# -------------------------
# Header
# -------------------------
render_navbar()
st.caption("Enter a title + body. Predictions are stored locally in SQLite (`data/predictions.sqlite3`).")


# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

label_filter = st.sidebar.selectbox("Label", ["All", "Real", "Fake"], index=0)
sources = ["All"] + fetch_sources()
source_filter = st.sidebar.selectbox("Source", sources, index=0)
min_conf = st.sidebar.slider("Min confidence (%)", 0, 100, 0, 5)

date_from = st.sidebar.text_input("Date from (YYYY-MM-DD)", value="")
date_to = st.sidebar.text_input("Date to (YYYY-MM-DD)", value="")
limit = st.sidebar.selectbox("Max rows", [10, 20, 50, 100], index=1)


# -------------------------
# Main layout
# -------------------------
left, right = st.columns([1.25, 0.85], vertical_alignment="top")

with left:
    st.subheader("Predict a News Article")

    with st.form("predict_form"):
        title = st.text_input("News Title", placeholder="Enter the headline/title...")
        body = st.text_area("News Body", height=220, placeholder="Paste the article body here...")
        source = st.text_input("News Source (optional)", placeholder="Reuters, CNN, BBC...")
        date_str = st.text_input("Date (optional)", placeholder="YYYY-MM-DD")
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not title.strip() and not body.strip():
            st.warning("Please provide at least a Title or Body.")
        else:
            out = predict(title=title, body=body, source=source, date_str=date_str)
            save_prediction(out)

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-title">Prediction: {label_badge(out['label'])}</div>
                  <div class="muted">Confidence: <b>{out['confidence_pct']:.2f}%</b></div>
                  <div class="muted">Category: <b>{out['category']}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{out['confidence_pct']:.2f}%")
            c2.metric("Prob Real", f"{out['prob_real'] * 100:.2f}%")
            c3.metric("Prob Fake", f"{out['prob_fake'] * 100:.2f}%")

            with st.expander("Raw probabilities"):
                st.json({"prob_fake": out["prob_fake"], "prob_real": out["prob_real"]})

with right:
    st.subheader("Overview")
    stats = fetch_stats()
    pct_real = (stats["real"] / stats["total"] * 100.0) if stats["total"] else 0.0

    a, b, c = st.columns(3)
    a.metric("Total", stats["total"])
    b.metric("Real", stats["real"])
    c.metric("Fake", stats["fake"])

    st.metric("% Real", f"{pct_real:.1f}%")
    st.metric("Avg Confidence", f"{stats['avg_conf']:.2f}%")

    st.info("Use the sidebar filters to explore saved predictions.")


# -------------------------
# Saved predictions
# -------------------------
st.divider()
st.subheader("Saved Predictions (Filtered)")

rows = fetch_filtered(
    label=label_filter,
    source=source_filter,
    min_conf=min_conf,
    date_from=(date_from.strip() or None),
    date_to=(date_to.strip() or None),
    limit=limit,
)

tab_all, tab_real, tab_fake = st.tabs(["All", "Real", "Fake"])

with tab_all:
    render_rows(rows)

with tab_real:
    render_rows([r for r in rows if (r.get("label") or "").lower() == "real"])

with tab_fake:
    render_rows([r for r in rows if (r.get("label") or "").lower() == "fake"])
