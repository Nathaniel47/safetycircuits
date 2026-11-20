"""SafetyCircuits feature browser: pick or type a prompt, see its refusal score and top features.

Runs from a precomputed cache, so it needs no GPU. Run: streamlit run app/browser.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from safetycircuits import browser_core as core  # noqa: E402

st.set_page_config(page_title="SafetyCircuits", layout="wide")
st.title("SafetyCircuits: how Gemma-2-2B-it decides to refuse")
st.caption(
    "Study of one small model, one direction and one set of sparse-autoencoder features. "
    "Refusal behaviour only: no harmful replies are shown or stored."
)

cache = core.load_cache()
tab_browser, tab_results = st.tabs(["Feature browser", "Study results"])


def show_prompt(entry: dict) -> None:
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=entry["score"],
                gauge={"axis": {"range": [-0.2, 1.4]}},
                title={"text": "Refusal score"},
            )
        )
        fig.update_layout(height=260, margin={"t": 60, "b": 0, "l": 20, "r": 20})
        st.plotly_chart(fig, width="stretch")
        st.write(f"Score {entry['score']:.2f}: {core.score_band(entry['score'])}.")
        if entry.get("refused") is not None:
            st.write("Model reply: **" + ("refused**" if entry["refused"] else "answered**"))
        st.caption(f"Group: {entry['group']}")
    with right:
        rows = core.feature_rows(entry, cache)
        labels = [f"#{r['feature']}" + (" (refusal)" if r["refusal_feature"] else "") for r in rows]
        fig = go.Figure(
            go.Bar(
                x=[r["activation"] for r in rows],
                y=labels,
                orientation="h",
                marker_color=["#c0392b" if r["refusal_feature"] else "#7f8c8d" for r in rows],
            )
        )
        fig.update_layout(
            title="Top SAE features at the last prompt token",
            height=340,
            yaxis={"autorange": "reversed"},
            margin={"t": 50, "b": 20},
        )
        st.plotly_chart(fig, width="stretch")
        links = [f"- feature {r['feature']}: [Neuronpedia]({r['url']})" for r in rows if r["url"]]
        st.markdown("\n".join(links))
    groups: dict[str, list[float]] = {}
    for p in cache["prompts"]:
        groups.setdefault(p["group"], []).append(p["score"])
    boxes = go.Figure()
    for name, scores in groups.items():
        boxes.add_trace(go.Box(x=scores, name=name, boxpoints=False, orientation="h"))
    boxes.add_vline(x=entry["score"], line_color="#c0392b", annotation_text="this prompt")
    boxes.update_layout(title="Where this prompt sits among the cached prompts", height=360)
    st.plotly_chart(boxes, width="stretch")


with tab_browser:
    if cache is None:
        st.info(
            "No cache yet. Build it on a GPU with `python -m safetycircuits.sae.build_cache` "
            "and put `browser_cache.json` in `results/`. See README.md."
        )
    else:
        meta = cache["meta"]
        st.write(
            f"Layer {meta['layer']} of Gemma-2-2B-it. **Refusal score**: 0 = like a typical "
            "harmless prompt, 1 = like a typical harmful one. Features come from Gemma Scope "
            "SAEs trained on the base model, so treat them as hints, not proof."
        )
        if "prompt_text" not in st.session_state:
            st.session_state["prompt_text"] = ""

        def pick(text: str) -> None:
            st.session_state["prompt_text"] = text

        st.write("Try an example:")
        cols = st.columns(4)
        for i, ex in enumerate(core.examples(cache)):
            cols[i % 4].button(
                f"{ex['group']}: {ex['text'][:38]}", key=f"ex{i}", on_click=pick, args=(ex["text"],)
            )
        text = st.text_input("Prompt", key="prompt_text")
        entry = core.find_prompt(cache, text) if text.strip() else None
        if text.strip() and entry is None:
            st.warning(
                "That prompt is not in the cache, and the live model is not available here. "
                f"Pick one of the {len(cache['prompts'])} cached prompts above, or run the "
                "model yourself on a GPU."
            )
        if entry:
            show_prompt(entry)

with tab_results:
    results = core.load_results()
    if not results:
        st.info("No result files in results/ yet.")
    if "direction" in results:
        st.subheader("Refusal direction (test split)")
        st.table(
            [
                {
                    "Prompts": name,
                    "Condition": cond,
                    "Refusal rate": f"{v['rate']:.1%}",
                    "n": v["n"],
                }
                for name, conds in results["direction"]["results"].items()
                for cond, v in conds.items()
            ]
        )
    if "jailbreaks" in results:
        summ = results["jailbreaks"]["summary"]
        st.subheader("Jailbreaks: share of harmful prompts answered instead of refused")
        fig = go.Figure(go.Bar(x=list(summ), y=[summ[n]["success_rate"] for n in summ]))
        fig.update_layout(yaxis_title="success rate", height=340)
        st.plotly_chart(fig, width="stretch")
    if "crosslingual" in results:
        langs = results["crosslingual"]["languages"]
        st.subheader("Does the English refusal direction work in other languages?")
        fig = go.Figure(
            go.Bar(x=list(langs), y=[v["all"]["direction_auroc"] for v in langs.values()])
        )
        fig.update_layout(yaxis_title="AUROC (0.5 = chance)", yaxis_range=[0, 1], height=320)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Check the comprehension control in results/crosslingual.json before reading Twi."
        )
