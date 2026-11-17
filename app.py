import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set page config
st.set_page_config(page_title="LLM Mechanics Playground", layout="wide")

st.title("🔬 How LLMs Work: Core Mechanics Playground")
st.caption("Explore Tokens, Context, and Sampling Parameters (Temperature, Top-K, Top-P, Repetition Penalty)")

# -------------------------------------------------------------
# 1. Load Model & Tokenizer (Cached so it loads only once)
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------------------------------------
# 2. Sidebar: Sampling Parameters & Context Window Limit
# -------------------------------------------------------------
st.sidebar.header("⚙️ Sampling Parameters")

temperature = st.sidebar.slider(
    "Temperature (T)",
    min_value=0.01,
    max_value=2.0,
    value=0.7,
    step=0.05,
    help="Scales logits: lower values make predictions sharp/deterministic; higher values flatten distribution (more random)."
)

top_k = st.sidebar.slider(
    "Top-K",
    min_value=0,
    max_value=100,
    value=40,
    step=1,
    help="Keeps only the top K most probable tokens; sets others to -infinity. Set to 0 to disable."
)

top_p = st.sidebar.slider(
    "Top-P (Nucleus Sampling)",
    min_value=0.05,
    max_value=1.0,
    value=0.9,
    step=0.05,
    help="Keeps the smallest set of tokens whose cumulative probability exceeds P."
)

repetition_penalty = st.sidebar.slider(
    "Repetition Penalty",
    min_value=1.0,
    max_value=2.5,
    value=1.1,
    step=0.05,
    help="Penalizes logits of tokens that already appeared in context to prevent looping."
)

st.sidebar.markdown("---")
st.sidebar.header("📦 Context Settings")

max_context_tokens = st.sidebar.slider(
    "Max Context Window (Tokens)",
    min_value=16,
    max_value=512,
    value=128,
    step=16,
    help="Limits the maximum number of recent tokens the model can see."
)

tokens_to_generate = st.sidebar.slider(
    "Tokens to Generate",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

# -------------------------------------------------------------
# 3. Custom Sampling Functions (Pure Math from Scratch)
# -------------------------------------------------------------
def apply_repetition_penalty(logits, past_token_ids, penalty):
    if penalty == 1.0 or not past_token_ids:
        return logits
    logits = logits.clone()
    for token_id in set(past_token_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits

def apply_temperature(logits, temp):
    return logits / temp

def apply_top_k(logits, k):
    if k <= 0 or k >= logits.size(-1):
        return logits
    top_k_values, _ = torch.topk(logits, k)
    min_val = top_k_values[-1]
    return torch.where(logits < min_val, torch.full_like(logits, -float("Inf")), logits)

def apply_top_p(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens outside nucleus
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits = logits.clone()
    logits[indices_to_remove] = -float("Inf")
    return logits

# -------------------------------------------------------------
# 4. Main App Interface
# -------------------------------------------------------------
prompt = st.text_area(
    "Enter Prompt:",
    value="Artificial Intelligence is",
    height=100
)

# Encode prompt into tokens (applies chat template for Instruct models if available)
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_token_ids = tokenizer.encode(formatted_prompt)
else:
    prompt_token_ids = tokenizer.encode(prompt)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Tokens Inspection")
    st.write(f"**Total Tokens:** `{len(prompt_token_ids)}`")
    # Display tokens as visual chips
    token_chips = [f"`{tokenizer.decode([tid])}` (ID: {tid})" for tid in prompt_token_ids]
    st.write(" **Token Breakdown:** " + " • ".join(token_chips))

with col2:
    st.subheader("2. Context Status")
    context_used = min(len(prompt_token_ids), max_context_tokens)
    st.progress(context_used / max_context_tokens)
    st.write(f"Context used: **{context_used} / {max_context_tokens} tokens**")
    if len(prompt_token_ids) > max_context_tokens:
        st.warning(f"⚠️ Prompt exceeds context limit! Truncated to the last {max_context_tokens} tokens.")

st.markdown("---")

# -------------------------------------------------------------
# 5. Generation Process
# -------------------------------------------------------------
if st.button("🚀 Generate Next Tokens", type="primary"):
    current_token_ids = list(prompt_token_ids)
    last_top_candidates = None

    with st.spinner("Generating step-by-step..."):
        output_box = st.empty()
        past_key_values = None
        for step in range(tokens_to_generate):
            # 1. Enforce Context Window (slice history to max_context_tokens)
            active_context_ids = current_token_ids[-max_context_tokens:]
            input_tensor = torch.tensor([active_context_ids])

            # 2. Get Model Logits for next token
            with torch.no_grad():
                if past_key_values is None:
                    outputs = model(input_tensor, use_cache=True)
                else:
                    outputs = model(
                        torch.tensor([[next_token]]),
                        past_key_values=past_key_values,
                        use_cache=True
                    )

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[0, -1, :]

            # 3. Apply Repetition Penalty
            logits_penalized = apply_repetition_penalty(next_token_logits, active_context_ids, repetition_penalty)

            # 4. Apply Temperature
            logits_temp = apply_temperature(logits_penalized, temperature)

            # 5. Apply Top-K
            logits_top_k = apply_top_k(logits_temp, top_k)

            # 6. Apply Top-P
            logits_top_p = apply_top_p(logits_top_k, top_p)

            # 7. Softmax to Probabilities
            probs = F.softmax(logits_top_p, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            stop_token_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
            if next_token in stop_token_ids:
               break
            current_token_ids.append(next_token)
            output_box.markdown(tokenizer.decode(current_token_ids[len(prompt_token_ids):]))

            # Record candidates for the final step to display
            if step == tokens_to_generate - 1:
                top_probs, top_indices = torch.topk(probs, k=min(10, (probs > 0).sum().item()))
                last_top_candidates = [
                    (tokenizer.decode([idx.item()]), float(prob.item() * 100))
                    for idx, prob in zip(top_indices, top_probs)
                ]

    st.subheader("📝 Generated Output")
    generated_text = tokenizer.decode(current_token_ids[len(prompt_token_ids):])
    st.success(generated_text)

    # ---------------------------------------------------------
    # 6. Top Candidates Probability Chart
    # ---------------------------------------------------------
    if last_top_candidates:
        st.subheader("📊 Next-Token Candidate Probabilities (Final Step)")
        words = [f"'{w}'" for w, p in last_top_candidates]
        probabilities = [p for w, p in last_top_candidates]
        st.bar_chart(dict(zip(words, probabilities)))
