"""The self-read reader: the SAME model interprets an activation from its own forward pass.

Mechanism:
  1) A small trainable Projection maps the cached hidden state h (R^d_in) into the
     model's token-embedding space -> k soft-token embeddings.
  2) We splice those soft tokens into a fixed chat template at a [SLOT] position and
     run the frozen model on inputs_embeds.
  3) We read the " Yes" vs " No" logits at the last position. Only the projection
     (and optionally LoRA) trains; the LLM weights are frozen.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

SENTINEL = "<<<STATE>>>"


class Projection(nn.Module):
    def __init__(self, d_in, d_model, k=1, hidden=0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, k * d_model)
            )
        else:
            self.net = nn.Linear(d_in, k * d_model)
        self.k, self.d = k, d_model

    def forward(self, h):                      # h: [B, d_in]
        return self.net(h).view(h.shape[0], self.k, self.d)


class Reader(nn.Module):
    def __init__(self, d_in, device="cuda", use_lora=None):
        super().__init__()
        use_lora = config.USE_LORA if use_lora is None else use_lora
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID, torch_dtype=torch.float32
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if use_lora:                            # optional fallback if projection underfits
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(
                self.model,
                LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                           lora_dropout=0.0, task_type="CAUSAL_LM"),
            )

        d_model = self.model.config.hidden_size
        self.proj = Projection(d_in, d_model, config.K_SOFT, config.PROJ_HIDDEN).to(device)

        # Build the slot template *via the real chat template* so the format is exact.
        msg = [{"role": "user", "content":
                f"Internal state: {SENTINEL}\nBased on the internal state above, is the "
                f"model's input an adversarial jailbreak prompt? Answer Yes or No."}]
        full = self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        pre, suf = full.split(SENTINEL)

        emb = self.model.get_input_embeddings()
        pre_ids = torch.tensor(self.tok(pre, add_special_tokens=False).input_ids, device=device)
        suf_ids = torch.tensor(self.tok(suf, add_special_tokens=False).input_ids, device=device)
        with torch.no_grad():
            self.pre_emb = emb(pre_ids).unsqueeze(0)   # [1, P, d]
            self.suf_emb = emb(suf_ids).unsqueeze(0)   # [1, S, d]

        self.yes_id = self.tok("Yes", add_special_tokens=False).input_ids[0]
        self.no_id = self.tok("No", add_special_tokens=False).input_ids[0]

    def forward(self, h):                       # h: [B, d_in] -> [B, 2] (Yes, No) logits
        b = h.shape[0]
        soft = self.proj(h).to(self.pre_emb.dtype)          # [B, k, d]
        pre = self.pre_emb.expand(b, -1, -1)
        suf = self.suf_emb.expand(b, -1, -1)
        inp = torch.cat([pre, soft, suf], dim=1)            # [B, P+k+S, d]
        attn = torch.ones(inp.shape[:2], device=self.device, dtype=torch.long)
        out = self.model(inputs_embeds=inp, attention_mask=attn)
        last = out.logits[:, -1, :]
        return last[:, [self.yes_id, self.no_id]]           # index 0 = Yes (=jailbreak)
