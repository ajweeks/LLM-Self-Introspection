"""Post-hoc analysis — pure CPU on cache/data.pt (+ results/predictions.json). No GPU.

Pulls the real findings and figures for the writeup:
  1) Baseline table w/ bootstrap 95% CI: length vs TF-IDF text vs hidden-state probe
     -> is the internal signal beating trivial length + shallow lexical features?
  2) Single-direction test (diff-of-means): how much of detection is one linear direction?
     (ties to Arditi et al. 2024 "refusal is mediated by a single direction")
  3) Length-overlap control: hidden-probe AUROC restricted to the length band both classes share
     -> is the win a length artifact?
  4) results/pca.png     : 2-D PCA of hidden states, colored by class
  5) results/length_hist.png : the length confound, shown honestly
  6) Error analysis on the trained self-read reader (top false pos / false neg)

Run:  python analyze.py   (after extract.py; error analysis also needs train.py)
"""
import os
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def auc_ci(y, s, n=1000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2:
            boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(boots)), float(lo), float(hi)


def probe_scores(Xtr, ytr, Xte):
    sc = StandardScaler(with_mean=False).fit(Xtr)   # with_mean=False -> works for sparse TF-IDF too
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def main():
    d = torch.load(config.CACHE, weights_only=False)
    os.makedirs("results", exist_ok=True)
    ytr, yte = d["train"]["y"].numpy(), d["test"]["y"].numpy()
    Ltr, Lte = d["train"]["len"].float().numpy(), d["test"]["len"].float().numpy()
    htr, hte = d["train"]["h"].numpy(), d["test"]["h"].numpy()
    txttr, txtte = d["train"]["text"], d["test"]["text"]
    report = {}

    # 1) baselines with bootstrap CI ------------------------------------------------
    s_len = probe_scores(Ltr.reshape(-1, 1), ytr, Lte.reshape(-1, 1))
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    s_tfidf = probe_scores(tf.fit_transform(txttr), ytr, tf.transform(txtte))
    s_hid = probe_scores(htr, ytr, hte)
    print("=== detection AUROC (test, 95% CI) ===")
    for name, s in [("length", s_len), ("tfidf_text", s_tfidf), ("hidden_state", s_hid)]:
        _, lo, hi = auc_ci(yte, s)
        a = roc_auc_score(yte, s)
        report[name] = {"auroc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"  {name:13s} {a:.3f}  [{lo:.3f}, {hi:.3f}]")

    # 2) single-direction (diff-of-means) -----------------------------------------
    direction = htr[ytr == 1].mean(0) - htr[ytr == 0].mean(0)
    direction /= np.linalg.norm(direction) + 1e-8
    a_dir = roc_auc_score(yte, hte @ direction)
    report["single_direction"] = {"auroc": round(a_dir, 4)}
    print(f"  single-direction (diff-of-means) {a_dir:.3f}  <- vs full probe {report['hidden_state']['auroc']:.3f}")

    # 3) length-overlap control ----------------------------------------------------
    lo_b, hi_b = np.percentile(Lte, [10, 90])
    mask = (Lte >= lo_b) & (Lte <= hi_b)
    if mask.sum() > 20 and len(np.unique(yte[mask])) == 2:
        a_band = roc_auc_score(yte[mask], s_hid[mask])
        report["hidden_length_band"] = {"auroc": round(a_band, 4),
                                        "band": [float(lo_b), float(hi_b)], "n": int(mask.sum())}
        print(f"  hidden within length band {lo_b:.0f}-{hi_b:.0f} tok: {a_band:.3f} (n={int(mask.sum())})")

    # 4) PCA scatter ---------------------------------------------------------------
    z = PCA(n_components=2).fit(htr).transform(hte)
    plt.figure(figsize=(5, 5))
    for lab, c, name in [(0, "tab:blue", "regular"), (1, "tab:red", "jailbreak")]:
        m = yte == lab
        plt.scatter(z[m, 0], z[m, 1], s=8, alpha=0.5, c=c, label=name)
    plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA of layer-{config.LAYER} hidden states (test)")
    plt.tight_layout(); plt.savefig("results/pca.png", dpi=120)
    print("saved results/pca.png")

    # 5) length histogram ----------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(Lte[yte == 0], bins=30, alpha=0.5, label="regular")
    plt.hist(Lte[yte == 1], bins=30, alpha=0.5, label="jailbreak")
    plt.xlabel("prompt length (tokens)"); plt.ylabel("count")
    plt.title("Length confound"); plt.legend()
    plt.tight_layout(); plt.savefig("results/length_hist.png", dpi=120)
    print("saved results/length_hist.png")

    # 6) error analysis on the trained self-read reader ----------------------------
    pf = "results/predictions.json"
    if os.path.exists(pf):
        pr = json.load(open(pf))
        y_sr, s_sr = np.array(pr["label"]), np.array(pr["self_read_score"])
        m, lo, hi = auc_ci(y_sr, s_sr)
        report["self_read"] = {"auroc": round(roc_auc_score(y_sr, s_sr), 4), "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"\n  self_read (HEADLINE)  {roc_auc_score(y_sr, s_sr):.3f}  95%CI [{lo:.3f}, {hi:.3f}]")
        rows = list(zip(pr["text"], pr["label"], pr["self_read_score"]))
        fps = sorted([r for r in rows if r[1] == 0], key=lambda r: -r[2])[:5]  # regular, high jailbreak score
        fns = sorted([r for r in rows if r[1] == 1], key=lambda r: r[2])[:5]   # jailbreak, low jailbreak score
        print("\n=== self-read top false positives (regular -> flagged) ===")
        for t, _, s in fps:
            print(f"  P(jb)={s:.2f} | {t[:90]!r}")
        print("=== self-read top false negatives (jailbreak -> missed) ===")
        for t, _, s in fns:
            print(f"  P(jb)={s:.2f} | {t[:90]!r}")
        report["self_read_errors"] = {
            "false_pos": [{"score": round(s, 3), "text": t[:200]} for t, _, s in fps],
            "false_neg": [{"score": round(s, 3), "text": t[:200]} for t, _, s in fns],
        }
    else:
        print("\n(no results/predictions.json yet -> run train.py for self-read error analysis)")

    json.dump(report, open("results/analysis.json", "w"), indent=2)
    print("\nsaved results/analysis.json")


if __name__ == "__main__":
    main()
