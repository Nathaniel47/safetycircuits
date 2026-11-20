# PORTFOLIO.md

## Portfolio card (about 120 words)

**SafetyCircuits: Interpreting Refusal and Jailbreaks in a Small LLM.** I studied how Google's Gemma-2-2B-it decides to refuse harmful requests. Removing a single direction in its internal activations cut refusals on harmful prompts from 98% to 8%, while removing a random direction changed nothing. I compared eight jailbreak styles, tested sparse-autoencoder features, and showed the same direction works in French and Swahili. For Twi the test was inconclusive: the model barely understands it. I built an interactive browser for exploring prompts and features, and wrote up the weak points honestly: one model, mostly one seed, unverified reply labels. Everything ran on free GPUs, and the demo needs none. Python, PyTorch, Hugging Face, Streamlit, Plotly. Code and write-up are open source.

Links to add: demo (after deploying, see DEPLOY.md), repository https://github.com/Nathaniel47/safetycircuits, write-up `docs/technical_writeup.md`.

## One sentence for emails to professors

> I recently built SafetyCircuits, an open study that reproduces the finding that refusal in a chat model is mediated by a single direction (Arditi et al., 2024) on Gemma-2-2B-it, adds Gemma Scope sparse-autoencoder features and a jailbreak comparison, and tests whether the mechanism carries over to French, Swahili and Twi, where I found the tools to be limited by how well the model understands the language, which relates to your work on interpretability and trustworthy AI.

**Adapting it.** The last clause ("which relates to your work on ...") is generic on purpose. Before sending, replace it with one specific paper or topic from that professor's own page, which I have not checked for RIT, Cornell, Texas Tech, Stevens InfinityLab or the Guelph AI Security Lab. Do not claim a link you have not read.

**Shorter version (about 30 words):** "My project SafetyCircuits tests whether the refusal direction in Gemma-2-2B-it, and the features behind it, carry over to jailbreaks and to low-resource languages such as Twi."
