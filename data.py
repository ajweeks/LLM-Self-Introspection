"""Load, dedup, balance and split the in-the-wild jailbreak dataset.

Adversarial = TrustAIRLab jailbreak prompts; benign = the same dataset's "regular"
prompts (same platforms/style -> better matched than an unrelated instruction set).
"""
import random
from datasets import load_dataset
import config

_DS = "TrustAIRLab/in-the-wild-jailbreak-prompts"


def _clean(ds):
    seen, out = set(), []
    for p in ds["prompt"]:
        if not p:
            continue
        p = p.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def get_splits(n_per_class=None, seed=None):
    n_per_class = n_per_class or config.N_PER_CLASS
    seed = config.SEED if seed is None else seed

    jb = _clean(load_dataset(_DS, "jailbreak_2023_12_25", split="train"))
    rg = _clean(load_dataset(_DS, "regular_2023_12_25", split="train"))

    rng = random.Random(seed)
    rng.shuffle(jb)
    rng.shuffle(rg)
    n = min(n_per_class, len(jb), len(rg))
    jb, rg = jb[:n], rg[:n]
    print(f"[data] using {n} jailbreak + {n} regular prompts")

    def split(items, label):          # 85/15 train/test (no val: no early-stopping in this build)
        a = int(0.85 * len(items))
        return {
            "train": [(t, label) for t in items[:a]],
            "test":  [(t, label) for t in items[a:]],
        }

    sj, sr = split(jb, 1), split(rg, 0)
    out = {}
    for s in ("train", "test"):
        d = sj[s] + sr[s]
        rng.shuffle(d)
        out[s] = d
    return out


if __name__ == "__main__":
    s = get_splits()
    for k, v in s.items():
        pos = sum(l for _, l in v)
        print(k, len(v), "examples,", pos, "jailbreak")
