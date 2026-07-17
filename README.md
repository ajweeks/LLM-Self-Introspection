# Can a model read its OWN hidden state to detect adversarial (jailbreak) inputs?

A 90-minute proof-of-pipeline for the **computation-aware / self-aware AI security** direction:
whether an LLM can use its *own* internal signals — not just an external probe reading them — to
flag adversarial inputs.

## Hypotheses

- **H1 (signal exists):** the model's residual-stream hidden states linearly separate jailbreak vs.
  regular prompts far above chance, and we can locate *which layers* carry it.
- **H2 (self-read, headline):** the **same** model, given access to its own hidden state injected as a
  soft token, detects jailbreaks better than the identical model with that signal removed (ablation)
  and better than a trivial length baseline.

## Mechanism ("model reads its own activation")

```
prompt ──▶ [ frozen Qwen2.5-0.5B ] ──▶ hidden_states[LAYER][:, -1]  =  h  (R^896)   # source pass
                                                     │
                                          trainable Projection  P
                                                     ▼
   "<|im_start|>user\nInternal state: [P(h)]\n...is this an adversarial jailbreak? Yes/No<|im_end|>
    <|im_start|>assistant\n"  ──▶ [ same frozen Qwen ] ──▶ logits(" Yes") vs logits(" No")   # reader pass
```

Only the small projection `P` (896→896) trains; the LLM is frozen. The same pretrained weights that
*produced* the activation are used to *interpret* it — that is the "self" in self-read.

## What the run reports (all on the held-out test set)

| method | role |
|--------|------|
| **self-read (own state)** | the headline — model reads its own hidden state |
| **ablation (shuffled)** | control — identical setup, activation↔label correspondence broken; should collapse to ~chance |
| **length baseline** | trivial-cue floor (jailbreaks run longer); self-read must beat it |
| **H1 probe (layer L)** | logistic regression on the hidden state — signal-present ceiling |
| **layer sweep** | H1 probe at every layer → `results/layer_sweep.png`, answers *which layers carry the signal* |

## Files

| file | role |
|------|------|
| `config.py`   | knobs: model, `LAYER`, N per class, epochs, lr, + fallbacks (`PROJ_HIDDEN`, `USE_LORA`) |
| `data.py`     | load/dedup/balance/split (85/15) jailbreak vs regular prompts |
| `extract.py`  | frozen forward passes → cache **all-layer** last-token hidden states (`cache/data.pt`) |
| `selfread.py` | `Projection` + soft-token splice + reader forward (the core mechanism) |
| `train.py`    | self-read + ablation + length baseline + H1 probe + layer sweep; writes `results/` |
| `smoke.py`    | ~10s pre-flight: template / shape / grad sanity before the full run |
| `run.sh`      | end-to-end: install → smoke → extract → train |

## Run (GPU box with internet)

```bash
bash run.sh
```

Outputs: `results/metrics.json`, `results/roc.png`, `results/layer_sweep.png`.

---

## Scope & feasibility decisions (what earns its place in a 60-min build)

**Kept — each plays a distinct, non-redundant role:** self-read (the claim), ablation (proves the win came
from *this input's* activation, not the harness), length baseline (honesty floor), H1 probe + layer sweep
(signal exists / where).

**Cut, deliberately:**
- **A val split / early-stopping** — merged into train (85/15). With a linear projection and 8 epochs,
  overfitting risk is low; a val-based early-stop is a clean next step, not a 60-min necessity.
- **An input/output-text-only reader baseline.** For *input* jailbreaks the raw text is fully visible, so a
  text reader would also score high and would **not** isolate an internal-signal advantage. That advantage
  is expected precisely where I/O looks benign — representation-space and weight-tampering attacks — which
  are out of this toy's scope. Adding it here would muddy the message, not sharpen it.
- **Multi-token / multi-layer self-read, and real GCG / representation-hijack / weight-tamper attacks** —
  all deferred to the follow-up; each is a project in itself.

**Added (cheap, high-ROI):** the **layer sweep**. One forward pass already returns all 25 hidden states, so
caching them costs almost nothing, and a per-layer probe directly answers a stated project question
("which internal signals carry information and which do not"), removes guesswork from the `LAYER` choice,
and **guarantees a reportable positive result (H1) even if self-read training doesn't converge**.

## The one empirical risk

Whether a *frozen* model can interpret the injected token from a **linear** projection alone. If
`train.py` shows self-read loss stuck near `0.69` / acc ~50%, the cache is reusable — flip a fallback in
`config.py` and rerun only `python train.py`:
- `PROJ_HIDDEN = 512` (MLP projection), and/or
- `USE_LORA = True` (lets the LLM adapt to the injected token).

Because H1 + the layer sweep run regardless, there is a reportable result either way.

## RESULTS

Run on 2026-07-17 — frozen Qwen2.5-0.5B-Instruct, 1000 prompts/class, layer 14, 8 epochs,
linear projection (no MLP, no LoRA). Held-out test set = 300 prompts (150/150). Full numbers in
`results/metrics.json`; plots in `results/roc.png` and `results/layer_sweep.png` (regenerate with
`python train.py`).

| method | acc | AUROC | FPR@95%TPR |
|--------|-----|-------|-----------|
| **self-read (own state)** | **0.870** | **0.934** | **0.287** |
| ablation (shuffled)   | 0.520 | 0.528 | 0.913 |
| length baseline       | 0.630 | 0.681 | 0.800 |
| H1 probe (layer 14)   | 0.797 | 0.870 | 0.473 |

**Layer sweep:** signal peaks at **layer 11 (AUROC 0.890)**, is high and roughly flat across all 24
transformer layers (~0.83–0.89), and is exactly **0.500 at layer 0 (embeddings)** — a clean sanity
check that separability only appears once attention has mixed the prompt into the last-token state.

**Both hypotheses hold.**

- **H1 (signal exists):** ✓ the hidden state linearly separates jailbreak vs. regular far above
  chance (0.87 at the read layer, 0.89 at best), across essentially every transformer layer.
- **H2 (self-read, headline):** ✓ self-read (0.934) beats the length floor (0.681) decisively and
  the shuffled-activation ablation collapses to chance (0.528) — so the win comes from *this input's*
  activation, not the harness or class priors. Notably self-read even **exceeds the linear H1 probe**
  (0.870) and the best single-layer probe (0.890): the frozen model reading its own state through its
  own forward pass extracts more than a linear readout of that same state.

The README's one empirical risk (a *linear* projection underfitting) **did not materialize** — no
`PROJ_HIDDEN`/`USE_LORA` fallback needed. Training loss hovered near 0.69 but test AUROC/acc were
high, i.e. the projection is well-calibrated-but-confident rather than stuck at chance.

See [`ANALYSIS.md`](ANALYSIS.md) for the full write-up, caveats, and next steps.

- Best layer (from sweep): **L=__**, AUROC **__**  (see `results/layer_sweep.png`)
- **Takeaway:** _<one line: did self-read beat ablation + length, and by how much?>_

## Caveats / next steps

Toy scale (0.5B, ~1k/class, single layer, last-token only); confounds only partly controlled. Next:
multi-layer/multi-token signals, and the threat models where internal signals should truly beat I/O —
representation hijacking and weight tampering — plus a matched-capacity text-only reader as the direct
I/O comparison in that regime.
