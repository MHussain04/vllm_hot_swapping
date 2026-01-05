import json
from transformers import AutoTokenizer
import numpy as np

# Load the dataset
dataset_path = "../datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

print("Loading dataset...")
with open(dataset_path, 'r') as f:
    data = json.load(f)

print(f"Total conversations: {len(data)}")

# Load tokenizer for Qwen3-0.6B
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

# Calculate token lengths for each conversation
print("Calculating token lengths...")
token_lengths = []

for i, conv in enumerate(data):
    # Combine all messages in the conversation
    full_text = ""
    if "conversations" in conv:
        for msg in conv["conversations"]:
            if "value" in msg:
                full_text += msg["value"] + " "
    
    # Tokenize
    tokens = tokenizer.encode(full_text)
    token_lengths.append(len(tokens))
    
    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1}/{len(data)} conversations...")

# Statistics
token_lengths = np.array(token_lengths)

print("\n" + "="*50)
print("ShareGPT Dataset Token Length Analysis")
print("="*50)
print(f"Total conversations: {len(token_lengths)}")
print(f"\nToken Length Statistics:")
print(f"  Min:    {np.min(token_lengths):,}")
print(f"  Max:    {np.max(token_lengths):,}")
print(f"  Mean:   {np.mean(token_lengths):,.2f}")
print(f"  Median: {np.median(token_lengths):,.2f}")
print(f"  Std:    {np.std(token_lengths):,.2f}")

print(f"\nPercentiles:")
for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
    print(f"  P{p}:   {np.percentile(token_lengths, p):,.0f}")

print(f"\nContext utilization analysis (32K context):")
print(f"  Conversations > 1K tokens:  {np.sum(token_lengths > 1000):,} ({100*np.sum(token_lengths > 1000)/len(token_lengths):.2f}%)")
print(f"  Conversations > 2K tokens:  {np.sum(token_lengths > 2000):,} ({100*np.sum(token_lengths > 2000)/len(token_lengths):.2f}%)")
print(f"  Conversations > 4K tokens:  {np.sum(token_lengths > 4000):,} ({100*np.sum(token_lengths > 4000)/len(token_lengths):.2f}%)")
print(f"  Conversations > 8K tokens:  {np.sum(token_lengths > 8000):,} ({100*np.sum(token_lengths > 8000)/len(token_lengths):.2f}%)")
print(f"  Conversations > 16K tokens: {np.sum(token_lengths > 16000):,} ({100*np.sum(token_lengths > 16000)/len(token_lengths):.2f}%)")
print(f"  Conversations > 32K tokens: {np.sum(token_lengths > 32000):,} ({100*np.sum(token_lengths > 32000)/len(token_lengths):.2f}%)")

print(f"\nAverage context utilization: {100*np.mean(token_lengths)/32768:.2f}% of 32K")