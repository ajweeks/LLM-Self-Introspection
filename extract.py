"""Frozen forward passes -> cache the model's own hidden state per prompt.

For each prompt we run the model on the prompt formatted as a user turn (with the
assistant generation prefix appended, i.e. exactly how the model would see it in
deployment) and grab the residual-stream vector at layer LAYER, last-token position.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from data import get_splits


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
    tok.padding_side = "left"                       # last token sits at index -1 for all rows
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.float32
    ).to(device).eval()

    def trunc(t):
        ids = tok(t, add_special_tokens=False).input_ids[: config.MAX_LEN]
        return tok.decode(ids)

    splits = get_splits()
    cache = {"layer": config.LAYER, "model": config.MODEL_ID}

    for name, items in splits.items():
        texts = [t for t, _ in items]
        labels = [l for _, l in items]
        H, L = [], []
        for i in range(0, len(texts), config.BATCH):
            batch = texts[i:i + config.BATCH]
            chat = [
                tok.apply_chat_template(
                    [{"role": "user", "content": trunc(t)}],
                    tokenize=False, add_generation_prompt=True,
                )
                for t in batch
            ]
            enc = tok(chat, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            out = model(**enc, output_hidden_states=True)
            # last-token vector at EVERY layer (embeddings + 24 blocks) -> [B, nL, d].
            # One pass already returns all of them, so the layer sweep is free.
            last_all = torch.stack([hs[:, -1, :] for hs in out.hidden_states], dim=1)
            H.append(last_all.float().cpu())
            L.extend(enc["attention_mask"].sum(1).cpu().tolist())
        h_all = torch.cat(H)                       # [N, nL, d]
        cache[name] = {
            "h_all": h_all,
            "h": h_all[:, config.LAYER, :],        # the layer self-read uses
            "y": torch.tensor(labels),
            "len": torch.tensor(L),
            "text": texts,
        }
        print(f"[extract] {name}: h_all {tuple(h_all.shape)}")

    os.makedirs(os.path.dirname(config.CACHE), exist_ok=True)
    torch.save(cache, config.CACHE)
    print("[extract] saved", config.CACHE)


if __name__ == "__main__":
    main()
