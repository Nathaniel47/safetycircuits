# PORTFOLIO.md

## Portfolio card (about 120 words)

**SafetyCircuits: Interpreting Refusal and Jailbreaks in a Small LLM.** I studied how Google's Gemma-2-2B-it decides to refuse harmful requests, using a refusal direction in its internal activations, sparse-autoencoder features, jailbreak prompts and three languages. My automatic judge suggested that removing the direction nearly eliminated refusals. A human check of 20 prompts showed otherwise: 15 were still refused and 5 complied, while the judge counted only 3 refusals. I published the correction. The direction still separates harmful from harmless prompts in French and Swahili; Twi was inconclusive. I built an interactive browser that needs no GPU. Python, PyTorch, Hugging Face, Streamlit, Plotly. Code and write-up are open source.

Links: demo https://safetycircuits.streamlit.app, repository https://github.com/Nathaniel47/safetycircuits, write-up `docs/technical_writeup.md`.

## One sentence for emails to professors

> I recently built SafetyCircuits, an open study that tests on Gemma-2-2B-it the finding that refusal in a chat model is mediated by a single direction (Arditi et al., 2024; a small human check showed my automatic judge had greatly overstated the effect), adds Gemma Scope sparse-autoencoder features and a jailbreak comparison, and tests whether the mechanism carries over to French, Swahili and Twi, where I found the tools to be limited by how well the model understands the language, which relates to your work on interpretability and trustworthy AI.

**Adapting it.** The last clause ("which relates to your work on ...") is generic on purpose. Before sending, replace it with one specific paper or topic from that professor's own page, which I have not checked for RIT, Cornell, Texas Tech, Stevens InfinityLab or the Guelph AI Security Lab. Do not claim a link you have not read.

## Tailored variants (one sentence each)

How these were made: I searched the web for each group (search summaries only; I did not read the lab pages in full, and you did not give me professor names). Each sentence uses only a topic I found stated by a source or that you listed yourself. Before sending, open the source, pick the specific paper you will mention, and replace `[Prof.]` with the name.

- **Guelph AI Security Lab** ([lab site](https://aisecurity.uoguelph.ca); described as covering security and privacy of generative models and defences against prompt injection for LLMs). *"Dear [Prof.], I built SafetyCircuits, an open study of how Gemma-2-2B-it decides to refuse harmful requests (one refusal direction, jailbreaks and low-resource languages), and I would value your view on how its lessons on measuring jailbreaks relate to your lab's work on securing generative models."*
- **Stevens InfinityLab** ([lab site](https://sites.google.com/stevens.edu/infinitylab); described as working on trustworthy machine learning, including trustworthy models, and on cybersecurity and safety). *"Dear [Prof.], my open project SafetyCircuits examines how a small language model's refusal mechanism behaves under jailbreak prompts and in low-resource languages such as Twi, which seems close to the trustworthy-model questions your lab works on."*
- **RIT** (an RIT news item describes an NSF CAREER project on defending ML models through interpretability and attribution: [news](https://www.rit.edu/news/professor-earns-nsf-career-award-defend-ai-models-attackers); I found no RIT group focused on LLM refusal). *"Dear [Prof.], I recently tested the finding that refusal in a chat model is mediated by a single direction, studying jailbreak prompts and sparse-autoencoder features, and learned from a human check how far an automatic judge can mislead, which I think connects to interpretability-based defences of ML models."*
- **Cornell (responsible AI, LLM evaluation: topic from your brief; I found no single Cornell group).** *"Dear [Prof.], SafetyCircuits is an open evaluation of how a small LLM refuses harmful requests, with baselines, seed variance and a human check showing where the refusal judge fails, and I would welcome feedback on its evaluation design."*
- **Texas Tech (trustworthy AI: topic from your brief; I found no matching lab).** *"Dear [Prof.], I built an open study of refusal behaviour in Gemma-2-2B-it, including a cross-lingual test whose Twi result turned out to be limited by the model's comprehension, and I am interested in how you think about trustworthiness across languages."*

Do not send any of these without checking that the link between your project and the professor's work is real and specific.

**Shorter version (about 30 words):** "My project SafetyCircuits studies how the refusal direction in Gemma-2-2B-it, and the features behind it, behave under jailbreaks and in low-resource languages such as Twi."
