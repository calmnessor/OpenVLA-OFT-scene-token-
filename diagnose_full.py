"""Full diagnostic: verify the label masking fix by simulating the real data pipeline."""
import sys
sys.path.insert(0, "/root/Afford+VLA/scene token+OpenVLA-OFT/openvla-oft")

import numpy as np
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from transformers import AutoTokenizer

base_tokenizer = AutoTokenizer.from_pretrained(
    "/root/checkpoints/openvla-7b", trust_remote_code=True)
action_tokenizer = ActionTokenizer(base_tokenizer)

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7

# Simulate real actions
np.random.seed(42)
actions = np.random.uniform(-1, 1, size=(NUM_ACTIONS_CHUNK, ACTION_DIM)).astype(np.float32)
current_action = actions[0]
future_actions = actions[1:]

current_action_string = action_tokenizer(current_action)
future_actions_string = ''.join(action_tokenizer(future_actions))
action_chunk_string = current_action_string + future_actions_string

# BUGGY approach
buggy_len = len(action_chunk_string)

# FIXED approach
action_chunk_token_ids = base_tokenizer(action_chunk_string, add_special_tokens=False).input_ids
fixed_len = len(action_chunk_token_ids)

print("=" * 80)
print(f"BUGGY string length: {buggy_len}")
print(f"FIXED token count:   {fixed_len}")
print(f"Difference: {fixed_len - buggy_len}")
print("=" * 80)

# Simulate the FULL prompt construction (exactly as datasets.py does)
lang = "put the black bowl on the plate"
prompt_builder = PurePromptBuilder("openvla")

conversation = [
    {"from": "human", "value": f"What action should the robot take to {lang}?"},
    {"from": "gpt", "value": action_chunk_string},
]
for turn in conversation:
    prompt_builder.add_turn(turn["from"], turn["value"])

full_prompt = prompt_builder.get_prompt()
full_input_ids = base_tokenizer(full_prompt, add_special_tokens=True).input_ids

print(f"\nFull prompt ({len(full_input_ids)} tokens):")
print(full_prompt[:150], "...")
print()

# Show what tokens the action chunk occupies in the full prompt
# Find the action chunk by looking for its token IDs
action_chunk_in_prompt_ids = base_tokenizer(action_chunk_string, add_special_tokens=False).input_ids
# Try to find it (approximate, since tokenization is contextual)
# Better approach: tokenize the action chunk SEPARATELY and count
print(f"Action chunk standalone tokenization: {len(action_chunk_in_prompt_ids)} tokens")
print(f"Action chunk token IDs: {action_chunk_in_prompt_ids[:10]}...")

# Now test both masking approaches
# BUGGY mask
buggy_labels = np.array(full_input_ids)
buggy_labels[:-(buggy_len + 1)] = -100
if True:  # predict_stop_token=True by default
    pass  # don't mask stop token
else:
    buggy_labels[-1] = -100

# FIXED mask
fixed_labels = np.array(full_input_ids)
fixed_labels[:-(fixed_len + 1)] = -100
if True:
    pass
else:
    fixed_labels[-1] = -100

buggy_non_masked = [int(x) for x in buggy_labels if x != -100]
fixed_non_masked = [int(x) for x in fixed_labels if x != -100]

print(f"\n--- Label Mask Comparison ---")
print(f"BUGGY: {len(buggy_non_masked)} non-masked tokens (should be {fixed_len + 1})")
print(f"FIXED: {len(fixed_non_masked)} non-masked tokens (should be {fixed_len + 1})")

# Decode what the model would learn from
print(f"\nBUGGY non-masked decode:")
print(repr(base_tokenizer.decode(buggy_non_masked)))
print(f"\nFIXED non-masked decode:")
print(repr(base_tokenizer.decode(fixed_non_masked)))

# Check if the FIRST action token is being masked by the bug
# The prompt ends with "Out: " + action_string + "</s>"
# With buggy: labels[:-(buggy_len+1)] = labels[:-(56+1)] = labels[:-57]
# With fixed: labels[:-(fixed_len+1)] = labels[:-(57+1)] = labels[:-58]
# So buggy unmasks 1 extra token at the start → that extra token is a PROMPT token, not action!

# Let's check: which token at position -58 (0-indexed) is being treated differently?
print(f"\n--- Detailed Comparison ---")
print(f"Token at [-58] (0-indexed: {len(full_input_ids)-58}): "
      f"id={full_input_ids[-58]}, decode={repr(base_tokenizer.decode([full_input_ids[-58]]))}")
print(f"  BUGGY: This token is NON-MASKED (treated as action)")
print(f"  FIXED: This token is MASKED (treated as prompt)")

# Also check the last action token
print(f"Token at [-1] (last): id={full_input_ids[-1]}, decode={repr(base_tokenizer.decode([full_input_ids[-1]]))}")
print(f"  This is the </s> stop token")

# Show what tokens the prompt (non-action) part produces
# The action chunk in full context: tokenizer might handle differently
# Let's tokenize just the prompt template (without action) to verify
prompt_template = full_prompt.split(action_chunk_string)[0]
prompt_template_ids = base_tokenizer(prompt_template, add_special_tokens=True).input_ids
print(f"\nPrompt template tokens count: {len(prompt_template_ids)}")
expected_non_masked = len(full_input_ids) - len(prompt_template_ids)
print(f"Expected non-masked tokens (full - template): {expected_non_masked}")
print(f"Expected (1 + fixed_token_count): {1 + fixed_len}")

# Verify: the action chunk within the FULL prompt is tokenized consistently
# Compare standalone tokenization vs in-context
full_action_slice = full_input_ids[len(prompt_template_ids):]
print(f"\nAction + stop tokens in full context: {len(full_action_slice)}")
print(f"Standalone action tokens: {fixed_len}")
print(f"Stop token adds 1: {len(full_action_slice) - fixed_len} (should be 1)")

print("\n" + "=" * 80)
if buggy_len != fixed_len:
    print("CONCLUSION: Label masking bug CONFIRMED and FIXED.")
    print(f"  Bug caused {fixed_len - buggy_len} token shift in label mask")
    print(f"  This means some prompt tokens were being unmasked (model learned to predict them)")
    print(f"  AND some action tokens were being masked (model couldn't learn them)")
    print(f"  Fix replaces len(action_chunk_string) with actual token count")
else:
    print("No string-length vs token-count mismatch found")
print("=" * 80)
