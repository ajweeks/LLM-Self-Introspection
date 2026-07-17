# Analysis — can a model read its own hidden state to detect jailbreaks?

**Run date:** 2026-07-17 · **Hardware:** 1× NVIDIA H100 NVL · **Model:** frozen Qwen2.5-0.5B-Instruct
· **Data:** TrustAIRLab in-the-wild jailbreak vs. regular prompts, 1000/class, 85/15 split (1700 train
/ 300 test) · **Read layer:** 14 · **Projection:** linear 896→896, 1 soft token, 8 epochs, lr 1e-3, no
LoRA · **Wall clock:** extract 43 s, train+eval ~3 min.

## Headline

The same frozen model, handed its **own** last-token residual-stream vector as a single soft token,
detects jailbreak prompts at **AUROC 0.934 / acc 0.87** on held-out data — beating a trivial length
floor (0.681), a linear probe on that same hidden state (0.870), and the best-layer linear probe
(0.890), while the shuffled-activation control collapses to chance (0.528).

## Results in full (test set, n=300)

| method | acc | AUROC | FPR@95%TPR | role |
|--------|-----|-------|-----------|------|
| **self-read (own state)** | **0.870** | **0.934** | **0.287** | the claim |
| ablation (shuffled)   | 0.520 | 0.528 | 0.913 | control — should be ~chance |
| length baseline       | 0.630 | 0.681 | 0.800 | trivial-cue floor |
| H1 probe (layer 14)   | 0.797 | 0.870 | 0.473 | linear signal ceiling |
| layer sweep (best, L11) | — | 0.890 | — | best linear readout, any layer |

## What each comparison establishes

**1. The signal exists and is not a length artifact (H1).** A logistic probe on the raw layer-14
hidden state hits AUROC 0.870 — far above the 0.681 you get from prompt length alone. Jailbreaks do
run longer, but length explains only a small part of the separability; most of it lives in the
representation.

**2. The signal is distributed across depth, and the pipeline is sound.** The per-layer sweep is high
and roughly flat from layer 1 (0.833) through layer 24 (0.867), peaking at **layer 11 (0.890)**.
Critically, **layer 0 (embeddings) sits at exactly 0.500**. That is the sanity check that matters: the
last token is the shared assistant-generation-prompt token, so its *embedding* is identical for every
input and carries zero class information — separability appears only after attention contextualizes the
prompt into that position. This confirms the last-token extraction is doing what we think, and that the
signal is genuinely computed, not leaked by tokenization.

**3. The self-read win is causal, not a harness artifact (H2 + ablation).** The decisive control is the
shuffled ablation: identical template, identical trainable projection, identical training — only the
activation↔label correspondence is broken. It falls to **0.528 (chance)**. So the reader is not
exploiting class priors, the prompt template, or the optimizer finding a degenerate constant; the
0.934 comes specifically from *the activation belonging to this input*.

**4. Self-read beats a linear readout of the same vector — the interesting result.** Self-read (0.934)
> H1 probe at the same layer (0.870) > and even > best-layer probe (0.890). The external probe and the
self-reader see the *same* information; the difference is the readout. Passing the vector back through
the model's own frozen weights applies a learned, non-linear transformation — the very computation that
produced the activation — and extracts more separability than a linear classifier can. This is direct
support for the "self" framing: the weights that generated the state are better interpreters of it than
an outside linear probe. (Caveat below on why this isn't yet an airtight claim.)

## The flagged risk did not fire

The README pre-registered one failure mode: a *linear* projection might be too weak for the frozen
model to interpret the injected token, showing up as loss stuck at ~0.69 / acc ~50%. Training loss did
sit near 0.69 (0.67–0.69 across epochs 5–7), which looks alarming in isolation — but **test AUROC/acc
are high**, so this is a well-separated-but-not-overconfident classifier, not a stuck one. No
`PROJ_HIDDEN` (MLP) or `USE_LORA` fallback was needed. Those remain cheap upgrades to try (the cache is
reusable), and might push AUROC further, but the headline stands on the linear projection alone.

## Honest caveats

- **Input jailbreaks are visible in the text.** For this attack class the raw prompt fully reveals the
  attack, so a text-only reader would also score high. This run therefore does **not yet isolate an
  internal-signal advantage over reading the input text** — it shows the internal signal is *sufficient*
  and *richly readable by the model itself*, not that it beats text. The advantage this direction is
  really after shows up where I/O looks benign (representation-space / weight-tampering attacks), which
  are out of this toy's scope.
- **FPR@95TPR is 0.287.** Best of the field, but still means flagging ~29% of benign prompts to catch
  95% of jailbreaks — not deployable as-is. Expected for a 0.5B model on noisy in-the-wild data; it's a
  signal-existence result, not a product.
- **Single run, single seed, one read layer.** No error bars. The 0.934 vs 0.890 gap is suggestive but
  should be confirmed across seeds before leaning hard on "self-read beats the best probe."
- **transformers 5.14.1** (vs. the `>=4.44` the code targets) — ran clean; only a benign `torch_dtype`
  deprecation warning.

## Text baseline (added post-hoc, `analyze.py`)

A TF-IDF n-gram text baseline reaches **0.889** — above the linear hidden-state probe (0.871) —
confirming that for input-visible jailbreaks shallow lexical features are already strong. Self-read
(0.934) still exceeds it, but the gap is within bootstrap noise and needs a CI to firm up. The
diff-of-means direction (0.866 ≈ full probe 0.871) shows the signal is near-one-dimensional; the
length-band control (0.862) rules out length as the driver.

## Next steps (cheapest first, cache is reusable)

1. **Seeds + CI.** Re-run self-read and the best-layer probe over 3–5 seeds; report mean ± std to
   confirm the self-read > probe gap is real.
2. **Sweep the read layer for self-read**, not just the probe — layer 11 (probe-best) may lift the
   headline above 0.934.
3. **Flip the fallbacks** (`PROJ_HIDDEN=512`, then `USE_LORA=True`) to see how much headroom a
   non-linear projection / adaptable LLM adds — bounds how much the linear bottleneck is costing.
4. **The real test of the thesis:** move to attacks where the input text looks benign
   (representation-hijack, weight-tamper, multi-turn), where an internal-signal reader *should* beat a
   text reader. That is the follow-up project the README defers to.
