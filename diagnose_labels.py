"""Diagnose the action_chunk_len=len(str) vs actual token count mismatch.

This verifies whether action_chunk_len = len(action_chunk_string) correctly identifies
the number of tokens the action chunk occupies in the tokenized prompt.
"""

import sys
sys.path.insert(0, "/root/Afford+VLA/scene token+OpenVLA-OFT/openvla-oft")

import numpy as np
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoTokenizer

# Load the same tokenizer used in training
base_tokenizer = AutoTokenizer.from_pretrained(
    "/root/checkpoints/openvla-7b",
    trust_remote_code=True,
)

action_tokenizer = ActionTokenizer(base_tokenizer)

print("=" * 80)
print("Action Tokenizer Diagnostic")
print("=" * 80)
print(f"Vocab size: {base_tokenizer.vocab_size}")
print(f"Action token begin idx: {action_tokenizer.action_token_begin_idx}")
print(f"Bins: {action_tokenizer.n_bins}")

# Simulate a realistic action chunk (8 actions × 7 dims = 56 action tokens, plus stop)
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7

# Generate realistic actions from [-1, 1]
np.random.seed(42)
actions = np.random.uniform(-1, 1, size=(NUM_ACTIONS_CHUNK, ACTION_DIM)).astype(np.float32)

current_action = actions[0]
future_actions = actions[1:]

current_action_string = action_tokenizer(current_action)
future_actions_string = ''.join(action_tokenizer(future_actions))
action_chunk_string = current_action_string + future_actions_string

# THE BUG: using string length
buggy_len = len(action_chunk_string)

# THE FIX: count actual token IDs
action_chunk_token_ids = base_tokenizer(action_chunk_string, add_special_tokens=False).input_ids
actual_token_len = len(action_chunk_token_ids)

print(f"\nAction chunk string length (BUGGY): {buggy_len}")
print(f"Action chunk token count (CORRECT): {actual_token_len}")
print(f"Difference: {actual_token_len - buggy_len}")
print(f"Mismatch? {'*** YES - BUG CONFIRMED ***' if buggy_len != actual_token_len else 'NO - no bug'}")

# Also print individual action token outputs
print(f"\n--- Individual action token details ---")
for i in range(min(3, ACTION_DIM)):
    dim_val = current_action[i]
    discretized = np.digitize(np.clip(dim_val, -1, 1), action_tokenizer.bins)
    token_id = base_tokenizer.vocab_size - discretized
    decoded_str = base_tokenizer.decode([token_id])
    retokenized_ids = base_tokenizer(decoded_str, add_special_tokens=False).input_ids
    print(f"  Dim {i}: val={dim_val:.4f} -> bin={discretized} -> token_id={token_id} -> "
          f'decode="{decoded_str}" (len={len(decoded_str)}) -> retokenized_ids={retokenized_ids} '
          f"(#tokens={len(retokenized_ids)}) {'***MISMATCH***' if len(retokenized_ids) != 1 else 'OK'}")

# Now test with the FULL prompt to see how many tokens the action chunk occupies
lang = "put the black bowl on the plate"
prompt_builder_class = None
# Try to import the prompt builder
from prismatic.models.backbones.llm.prompting import PromptBuilder
prompt_builder = PromptBuilder("openvla")

conversation = [
    {"from": "human", "value": f"What action should the robot take to {lang}?"},
    {"from": "gpt", "value": action_chunk_string},
]
for turn in conversation:
    prompt_builder.add_turn(turn["from"], turn["value"])

full_prompt = prompt_builder.get_prompt()
full_input_ids = base_tokenizer(full_prompt, add_special_tokens=True).input_ids
full_labels = list(full_input_ids)

print(f"\n--- Full prompt analysis ---")
print(f"Full prompt token count: {len(full_input_ids)}")
print(f"Prompt: {full_prompt[:200]}...")

# Show what would be masked with buggy_len
buggy_labels = list(full_input_ids)
buggy_labels[:- (buggy_len + 1)] = -100
print(f"\nWith BUGGY string length ({buggy_len}):")
print(f"  Non-masked (action) tokens: {sum(1 for x in buggy_labels if x != -100)}")
print(f"  Non-masked token IDs: {[x for x in buggy_labels if x != -100][:20]}...")

# Show what would be masked with correct token count
correct_labels = list(full_input_ids)
correct_labels[:- (actual_token_len + 1)] = -100
print(f"\nWith CORRECT token count ({actual_token_len}):")
print(f"  Non-masked (action) tokens: {sum(1 for x in correct_labels if x != -100)}")
print(f"  Non-masked token IDs: {[x for x in correct_labels if x != -100][:20]}...")

# Decode the non-masked tokens to see what the model is actually learning from
buggy_action_tokens = base_tokenizer.decode([x for x in buggy_labels if x != -100])
correct_action_tokens = base_tokenizer.decode([x for x in correct_labels if x != -100])
print(f"\nBuggy non-masked decode: {buggy_action_tokens[:200]}")
print(f"Correct non-masked decode: {correct_action_tokens[:200]}")

# Count how many tokens are in the prompt BEFORE the action chunk
# The prompt ends with "Out: " so let's count everything up to the action
prompt_only = prompt_builder.get_prompt().split(action_chunk_string)[0]
prompt_only_ids = base_tokenizer(prompt_only, add_special_tokens=True).input_ids
print(f"\nPrompt tokens BEFORE action chunk: {len(prompt_only_ids)}")
print(f"Action chunk tokens (full prompt minus preamble): {len(full_input_ids) - len(prompt_only_ids)}")
print(f"Expected (1 + action_chunk_token_len): {1 + actual_token_len}")
print(f"Expected (1 + buggy_len): {1 + buggy_len}")

print("\n" + "=" * 80)
print("CONCLUSION:")
if buggy_len != actual_token_len:
    print(f"  BUG CONFIRMED: String length ({buggy_len}) != token count ({actual_token_len})")
    print(f"  This causes the label mask to be shifted by {actual_token_len - buggy_len} tokens!")
    print(f"  Fix: Use actual token count instead of string length")
else:
    print(f"  No string-length vs token-count bug found for this tokenizer")
    print(f"  (But there may be other issues causing false convergence)")
print("=" * 80)
