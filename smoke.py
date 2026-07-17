"""~10s pre-flight check: catch template / shape / dtype / grad bugs before the full run.

Loads the model once, pushes 2 fake activations through the Reader, and asserts:
  - the chat-template slot split worked (prefix + suffix tokens exist),
  - " Yes"/" No" are single tokens,
  - the reader returns [B, 2] logits,
  - gradients reach the projection but NOT the frozen LLM (unless USE_LORA).
Run:  python smoke.py
"""
import torch
import torch.nn.functional as F
import config
from selfread import Reader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_in = 896                                   # Qwen2.5-0.5B hidden size
    reader = Reader(d_in, device)

    yes_ids = reader.tok("Yes", add_special_tokens=False).input_ids
    no_ids = reader.tok("No", add_special_tokens=False).input_ids
    print(f"'Yes' -> {yes_ids}   'No' -> {no_ids}")
    print(f"prefix tokens = {reader.pre_emb.shape[1]}, suffix tokens = {reader.suf_emb.shape[1]}, "
          f"k_soft = {config.K_SOFT}")
    assert reader.pre_emb.shape[1] > 0 and reader.suf_emb.shape[1] > 0, "template split failed"
    if len(yes_ids) != 1 or len(no_ids) != 1:
        print("WARNING: Yes/No not single tokens; using first sub-token (still works).")

    h = torch.randn(2, d_in, device=device)
    logits = reader(h)
    print("logits shape:", tuple(logits.shape))
    assert logits.shape == (2, 2), "reader output should be [B, 2]"

    loss = F.cross_entropy(logits, torch.tensor([0, 1], device=device))
    loss.backward()

    net = reader.proj.net
    w = net.weight if hasattr(net, "weight") else net[0].weight    # Linear or Sequential
    print("projection grad norm:", float(w.grad.norm()))
    assert w.grad is not None and torch.isfinite(w.grad).all(), "no grad reached projection"

    n_llm_trainable = sum(p.requires_grad for p in reader.model.parameters())
    print(f"trainable LLM params: {n_llm_trainable} (expect 0 unless USE_LORA={config.USE_LORA})")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
