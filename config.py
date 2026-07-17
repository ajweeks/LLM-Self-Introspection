"""Shared config for the self-read jailbreak-detection toy project."""

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # ungated, tiny, instruct-tuned. hidden=896, 24 layers.

# Which residual-stream layer to read the internal signal from.
# hidden_states has 25 entries: index 0 = embeddings, 1..24 = layer outputs.
LAYER = 14

K_SOFT = 1          # number of soft tokens the activation is projected into
N_PER_CLASS = 1000  # balanced cap per class (jailbreak / regular)
MAX_LEN = 512       # truncate prompts to this many tokens before templating

BATCH = 32
EPOCHS = 8
LR = 1e-3

# Fallbacks if the frozen linear projection underfits (loss stuck ~0.69, acc ~50%):
PROJ_HIDDEN = 0     # >0 makes the projection a 2-layer MLP with this hidden width
USE_LORA = False    # True adds LoRA (r=8) on q/v_proj so the LLM adapts to the injected token

SEED = 0
CACHE = "cache/data.pt"
