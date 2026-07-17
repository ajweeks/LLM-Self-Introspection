"""Train + evaluate the whole pipeline and write results.

Reports side-by-side on the held-out test set:
  - self-read           : model reads its OWN hidden state (the headline)
  - ablation (shuffled) : identical setup, activation<->label correspondence broken (control)
  - length baseline     : logistic regression on prompt token length (trivial-cue floor)
  - H1 probe (hidden)   : logistic regression directly on the hidden state (signal-present ceiling)
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from selfread import Reader


def fpr_at_tpr(y, score, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y, score)
    idx = int(np.argmax(tpr >= tpr_target))
    return float(fpr[idx])


def summarize(y, score, pred):
    return {
        "auroc": float(roc_auc_score(y, score)),
        "acc": float(accuracy_score(y, pred)),
        "fpr@95tpr": fpr_at_tpr(y, score),
    }


def sklearn_probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    return summarize(yte, s, (s >= 0.5).astype(int)), (yte, s)


def reader_run(d, device, shuffle_control):
    reader = Reader(d["train"]["h"].shape[1], device)
    trainable = [p for p in reader.parameters() if p.requires_grad]   # proj (+ LoRA if enabled)
    opt = torch.optim.Adam(trainable, lr=config.LR)
    ce = torch.nn.CrossEntropyLoss()

    def prep(split):
        h, y = d[split]["h"].float(), d[split]["y"].long()
        if shuffle_control:                          # break activation<->label correspondence
            g = torch.Generator().manual_seed(config.SEED + 1)
            h = h[torch.randperm(h.shape[0], generator=g)]
        return h, y

    ht, yt = prep("train")
    dl = DataLoader(TensorDataset(ht, yt), batch_size=config.BATCH, shuffle=True)
    for ep in range(config.EPOCHS):
        reader.proj.train()
        tot = 0.0
        for hb, yb in dl:
            hb, yb = hb.to(device), yb.to(device)
            logits = reader(hb)
            loss = ce(logits, 1 - yb)                # jailbreak(1) -> Yes (index 0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep} loss {tot/len(dl):.3f}")

    reader.proj.eval()
    hte, yte = prep("test")
    scores, preds = [], []
    with torch.no_grad():
        for i in range(0, hte.shape[0], config.BATCH):
            logits = reader(hte[i:i + config.BATCH].to(device))
            scores.append(torch.softmax(logits, 1)[:, 0].cpu())   # P(Yes = jailbreak)
            preds.append((logits[:, 0] > logits[:, 1]).long().cpu())
    score, pred, y = torch.cat(scores).numpy(), torch.cat(preds).numpy(), yte.numpy()
    return summarize(y, score, pred), (y, score), reader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(config.CACHE, weights_only=False)
    res, roc = {}, {}

    ytr, yte = d["train"]["y"].numpy(), d["test"]["y"].numpy()

    print("[H1] logistic probe on hidden state ...")
    res["h1_probe_hidden"], roc[f"H1 probe (layer {config.LAYER})"] = sklearn_probe(
        d["train"]["h"].numpy(), ytr, d["test"]["h"].numpy(), yte)

    print("[layer sweep] which layer carries the jailbreak signal ...")
    htr_all, hte_all = d["train"]["h_all"].numpy(), d["test"]["h_all"].numpy()
    aucs = [sklearn_probe(htr_all[:, L, :], ytr, hte_all[:, L, :], yte)[0]["auroc"]
            for L in range(htr_all.shape[1])]
    best = int(np.argmax(aucs))
    res["layer_sweep"] = {"auroc_per_layer": [round(a, 4) for a in aucs],
                          "best_layer": best, "best_auroc": round(aucs[best], 4)}
    print(f"  best layer = {best} (AUROC {aucs[best]:.3f}); using layer {config.LAYER} for self-read")

    print("[baseline] length ...")
    res["length_baseline"], roc["length baseline"] = sklearn_probe(
        d["train"]["len"].float().numpy().reshape(-1, 1), ytr,
        d["test"]["len"].float().numpy().reshape(-1, 1), yte)

    print("[self-read] model reads its own hidden state ...")
    res["self_read"], roc["self-read (own state)"], reader = reader_run(d, device, False)

    print("[control] ablation: shuffled activation ...")
    res["ablation_shuffle"], roc["ablation (shuffled)"], _ = reader_run(d, device, True)

    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/metrics.json", "w"), indent=2)
    print("\n=== metrics (test set) ===")
    print(json.dumps(res, indent=2))

    # per-example self-read test predictions (test order preserved) -> feeds analyze.py error analysis
    y_sr, s_sr = roc["self-read (own state)"]
    json.dump({"text": d["test"]["text"],
               "label": [int(v) for v in y_sr.tolist()],
               "self_read_score": [float(v) for v in s_sr.tolist()]},
              open("results/predictions.json", "w"))

    plt.figure(figsize=(6, 6))
    for name, (y, s) in roc.items():
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(y, s):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Jailbreak detection from the model's own hidden state")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("results/roc.png", dpi=120)
    print("saved results/roc.png")

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(aucs)), aucs, marker="o")
    plt.axvline(config.LAYER, ls="--", c="r", label=f"used for self-read (L={config.LAYER})")
    plt.axvline(best, ls=":", c="g", label=f"best (L={best})")
    plt.xlabel("layer (0 = embeddings)"); plt.ylabel("H1 probe AUROC")
    plt.ylim(0.45, 1.02); plt.title("Which layer carries the jailbreak signal")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("results/layer_sweep.png", dpi=120)
    print("saved results/layer_sweep.png")

    print("\n=== example self-read predictions ===")
    hte, yte, txt = d["test"]["h"].float(), d["test"]["y"], d["test"]["text"]
    reader.proj.eval()
    idxs = list(range(0, len(txt), max(1, len(txt) // 8)))[:8]
    with torch.no_grad():
        for i in idxs:
            logits = reader(hte[i:i + 1].to(device))
            p = torch.softmax(logits, 1)[0, 0].item()
            tag = "JB " if yte[i] == 1 else "reg"
            print(f"  {tag} P(jailbreak)={p:.2f} | {txt[i][:80]!r}")


if __name__ == "__main__":
    main()
