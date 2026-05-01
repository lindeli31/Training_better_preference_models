import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# -----------------------------
# Load model
# -----------------------------
model_name = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

model.eval()

# -----------------------------
# Original text
# -----------------------------
text = "The approach improves robustness while maintaining efficiency."

print("ORIGINAL:")
print(text)

# -----------------------------
# Tokenize
# -----------------------------
inputs = tokenizer(text, return_tensors="pt")

# -----------------------------
# Encoder forward pass
# -----------------------------
with torch.no_grad():
    encoder_outputs = model.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask
    )

hidden = encoder_outputs.last_hidden_state

print("\nHidden state shape:")
print(hidden.shape)

# Example:
# [batch_size, seq_len, hidden_dim]

# -----------------------------
# Add small perturbation
# -----------------------------
sigma = 0.08

noise = torch.randn_like(hidden) * sigma

perturbed_hidden = hidden + noise

# -----------------------------
# Decode ORIGINAL hidden states
# -----------------------------
with torch.no_grad():

    original_generated_ids = model.generate(
        encoder_outputs=BaseModelOutput(
            last_hidden_state=hidden
        ),
        max_new_tokens=50
    )

original_generated_text = tokenizer.decode(
    original_generated_ids[0],
    skip_special_tokens=True
)

# -----------------------------
# Decode PERTURBED hidden states
# -----------------------------
with torch.no_grad():

    perturbed_generated_ids = model.generate(
        encoder_outputs=BaseModelOutput(
            last_hidden_state=perturbed_hidden
        ),
        max_new_tokens=50
    )

perturbed_generated_text = tokenizer.decode(
    perturbed_generated_ids[0],
    skip_special_tokens=True
)

# -----------------------------
# Print outputs
# -----------------------------
print("\nRECONSTRUCTED FROM ORIGINAL LATENT:")
print(original_generated_text)

print("\nRECONSTRUCTED FROM PERTURBED LATENT:")
print(perturbed_generated_text)