"""SafetyCircuits feature browser: pick or type a prompt, see its refusal score and top features.

Runs from a precomputed cache, so it needs no GPU. Run: streamlit run app/browser.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from safetycircuits import browser_core as core  # noqa: E402
from safetycircuits.config import RESULTS_DIR  # noqa: E402

REPO = "https://github.com/Nathaniel47/safetycircuits"

st.set_page_config(page_title="SafetyCircuits", layout="wide")


@st.cache_resource(show_spinner=False)
def get_cache(path: str, mtime: float):
    """Parsed once per server process (and again if the file changes)."""
    return core.load_cache(Path(path))


@st.cache_resource(show_spinner=False)
def get_results(mtime: float):
    return core.load_results()


path = core.cache_path()
cache = get_cache(str(path), path.stat().st_mtime if path.exists() else 0.0)

st.title("SafetyCircuits: how Gemma-2-2B-it decides to refuse")
st.caption(
    "An open study of one small model: one refusal direction, sparse-autoencoder features, "
    "jailbreaks and three languages. Refusal behaviour only: no harmful replies are shown "
    "or stored."
)
tab_browser, tab_results, tab_about = st.tabs(["Feature browser", "Study results", "About"])


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
        labels = [
            f"feature {r['feature']}" + (" (refusal)" if r["refusal_feature"] else "") for r in rows
        ]
        fig = go.Figure(
            go.Bar(
                x=[r["lift"] if r["lift"] is not None else r["activation"] for r in rows],
                y=labels,
                orientation="h",
                marker_color=["#c0392b" if r["refusal_feature"] else "#7f8c8d" for r in rows],
            )
        )
        fig.update_layout(
            title="SAE features that fire most above their usual level (red = refusal feature)",
            height=340,
            yaxis={"autorange": "reversed", "type": "category"},
            margin={"t": 50, "b": 20},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Bar length = how far the feature is above its average on harmless prompts, at the "
            "last prompt token. Red bars are among the 20 features that best separate harmful from "
            "harmless prompts."
        )
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


def pick(text: str) -> None:
    st.session_state["prompt_text"] = text


def pick_from_list() -> None:
    chosen = st.session_state.get("list_prompt")
    if chosen:
        st.session_state["prompt_text"] = chosen


with tab_browser:
    if cache is None:
        st.info(
            "No cache yet. Build it on a GPU with `python -m safetycircuits.sae.build_cache` "
            "and put `browser_cache.json` in `results/`. See README.md."
        )
    else:
        meta = cache["meta"]
        st.write(
            f"Pick a prompt and see how strongly Gemma-2-2B-it (layer {meta['layer']}) treats it "
            "like a harmful request. **Refusal score**: 0 = like a typical harmless prompt, "
            "1 = like a typical harmful one. The features come from Gemma Scope sparse "
            "autoencoders trained on the base model, so treat them as hints, not proof."
        )
        if "prompt_text" not in st.session_state:
            st.session_state["prompt_text"] = ""
        st.write("**Try an example:**")
        cols = st.columns(3)
        for i, ex in enumerate(core.examples(cache, per_group=1)):
            cols[i % 3].button(
                f"{ex['group']}: {ex['text'][:40]}",
                key=f"ex{i}",
                on_click=pick,
                args=(ex["text"],),
                width="stretch",
            )
        with st.expander(f"Or browse all {len(cache['prompts'])} cached prompts"):
            group = st.selectbox("Group", core.group_names(cache), key="list_group")
            options = [e["text"] for e in core.prompts_in_group(cache, group)]
            st.selectbox("Prompt", options, key="list_prompt", index=None, on_change=pick_from_list)
        text = st.text_input(
            "Prompt (must be one of the cached prompts; the model is not running here)",
            key="prompt_text",
        )
        entry = core.find_prompt(cache, text) if text.strip() else None
        if text.strip() and entry is None:
            st.warning(
                "That prompt is not in the cache, because the model is not running in this demo. "
                "Pick one of the cached prompts instead."
            )
            for i, near in enumerate(core.suggest(cache, text)):
                st.button(
                    f"Did you mean: {near['text'][:70]}",
                    key=f"near{i}",
                    on_click=pick,
                    args=(near["text"],),
                )
        if entry:
            show_prompt(entry)

with tab_results:
    results = get_results(0.0)
    if not results:
        st.info("No result files in results/ yet.")
    st.write(
        "Numbers from one run of each experiment on Gemma-2-2B-it. Intervals are 95%. "
        f"The full table and the list of weaknesses are in the [repository]({REPO}) "
        "(`results/results_table.md` and `results/limitations.md`)."
    )
    if "direction" in results:
        st.subheader("1. Removing the refusal direction removes the refusals")
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
    for name, title, note in (
        (
            "chart_layer_sweep.png",
            "2. Which layer? Removal works best at early layers and layer 25",
            "Layers 8, 10, 12, 14 and 25 tie at 0% on validation prompts; only layers 24 and 25 "
            "were confirmed on the test split.",
        ),
        (
            "chart_jailbreaks.png",
            "3. Jailbreaks",
            "Grey bars are the baseline and two control suffixes. Success means the reply was not "
            "a refusal; it was not read by a person.",
        ),
        (
            "chart_crosslingual.png",
            "4. Other languages",
            "French and Swahili work. Twi is at chance, but Gemma barely understands Twi "
            "(see the chrF under each bar), so that is not a safety result.",
        ),
    ):
        image = RESULTS_DIR / name
        if image.exists():
            st.subheader(title)
            st.image(str(image), width=700)
            st.caption(note)

with tab_about:
    n_prompts = len(cache["prompts"]) if cache else 0
    st.markdown(
        f"""
**What this is.** A study of how Gemma-2-2B-it decides to refuse, built from public benchmark
prompts (AdvBench, HarmBench, XSTest, Alpaca-cleaned, FLORES-200).

**Method in four lines**
1. The refusal direction is the average difference between the model's internal activations on
   harmful and harmless prompts.
2. Removing it makes the model stop refusing; adding it makes it refuse harmless prompts.
3. Sparse autoencoders (Gemma Scope) split activations into features, to see what fires on
   refusals.
4. The same direction is tested on jailbreak prompts and on French, Swahili and Twi.

**What this demo is not.** The model is not running here. Scores and features were computed
once on a GPU for the {n_prompts} prompts you can pick, so only those can be shown. No harmful
replies are shown or stored.

**Read before you trust it.** One model, mostly one seed, a phrase-matching refusal check that
was not verified against harmful replies, and poor Twi translations. See
`results/limitations.md`.

Code, data notes and licences: [{REPO}]({REPO}). Gemma is used under the Gemma Terms of Use.
"""
    )
