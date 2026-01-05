#!/usr/bin/env python3
import json
import random
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import requests

# Initialize tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

def get_text_sources():
    """Load various text sources for creating long prompts"""
    texts = []
    
    print("Loading text sources...")
    
    # Using larger, more realistic document counts
    try:
        print("Attempting to load FineWeb-Edu dataset...")
        fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        fineweb_texts = []
        for i, item in enumerate(fineweb):
            if i >= 1000: break
            if 'text' in item and len(item['text']) > 1000:
                fineweb_texts.append(item['text'])
        texts.extend(fineweb_texts)
        print(f"Loaded {len(fineweb_texts)} FineWeb-Edu documents")
    except Exception as e:
        print(f"FineWeb-Edu not available: {e}")
    
    try:
        print("Attempting to load The Pile dataset...")
        pile = load_dataset("EleutherAI/pile", "all", split="train", streaming=True)
        pile_texts = []
        for i, item in enumerate(pile):
            if i >= 600: break
            if 'text' in item and len(item['text']) > 1000:
                pile_texts.append(item['text'])
        texts.extend(pile_texts)
        print(f"Loaded {len(pile_texts)} Pile documents")
    except Exception as e:
        print(f"The Pile not available: {e}")
    
    try:
        print("Attempting to load Gutenberg books...")
        book_urls = [
            "https://www.gutenberg.org/files/1342/1342-0.txt", "https://www.gutenberg.org/files/11/11-0.txt",
            "https://www.gutenberg.org/files/84/84-0.txt", "https://www.gutenberg.org/files/1661/1661-0.txt",
            "https://www.gutenberg.org/files/98/98-0.txt", "https://www.gutenberg.org/files/2701/2701-0.txt",
            "https://www.gutenberg.org/files/1400/1400-0.txt"
        ]
        for url in book_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    texts.append(response.text)
                    print(f"  Loaded book from {url.split('/')[-1]}")
            except: pass
    except Exception as e:
        print(f"Gutenberg books not available: {e}")
    
    try:
        print("Attempting to load CC-News dataset...")
        cc_news = load_dataset("cc_news", split="train", streaming=True)
        cc_texts = []
        for i, item in enumerate(cc_news):
            if i >= 400: break
            if 'text' in item and len(item['text']) > 1000:
                cc_texts.append(item['text'])
        texts.extend(cc_texts)
        print(f"Loaded {len(cc_texts)} CC-News articles")
    except Exception as e:
        print(f"CC-News not available: {e}")
    
    print(f"\nTotal: Loaded {len(texts)} text sources")
    return texts


def create_prompt_of_length(texts, target_tokens, tolerance=0.1):
    """Create a prompt with approximately target_tokens length and return prompt and token count."""
    
    if not texts:
        print("No texts available.")
        return "", 0

    random.shuffle(texts)
    
    prompt_parts = []
    current_tokens = 0
    min_tokens = int(target_tokens * (1 - tolerance))
    max_tokens = int(target_tokens * (1 + tolerance))
    
    # Step 1: Assemble a prompt that is roughly the right size
    for text in texts:
        if current_tokens >= min_tokens:
            break
        prompt_parts.append(text)
        if len(prompt_parts) % 5 == 0:
            combined = "\n\n".join(prompt_parts)
            current_tokens = len(tokenizer.encode(combined, truncation=False, add_special_tokens=False))
            if current_tokens > max_tokens:
                while current_tokens > max_tokens and len(prompt_parts) > 1:
                    prompt_parts.pop()
                    combined = "\n\n".join(prompt_parts)
                    current_tokens = len(tokenizer.encode(combined, truncation=False, add_special_tokens=False))
                break
    
    # Step 2: Add instruction prefix
    instruction_prefix = "Please analyze and summarize the following text:\n\n"
    final_prompt_text = instruction_prefix + "\n\n".join(prompt_parts)
    
    # Step 3: Tokenize and perform final, guaranteed truncation/padding
    token_ids = tokenizer.encode(final_prompt_text, truncation=False, add_special_tokens=False)
    
    # Pad up to the minimum if we are too short
    while len(token_ids) < min_tokens:
        padding_text = "\n\n... (additional content for length)"
        padding_tokens = tokenizer.encode(padding_text, add_special_tokens=False)
        token_ids.extend(padding_tokens)

    # Truncate to the maximum if we are too long (after assembly or padding)
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]

    # Decode back to a string for the final prompt
    final_prompt = tokenizer.decode(token_ids)
    final_token_count = len(token_ids)
    
    return final_prompt, final_token_count


def create_dataset(output_dir="datasets"):
    """Create the full dataset, with a separate file and stats for each prompt length."""
    
    os.makedirs(output_dir, exist_ok=True)
    texts = get_text_sources()
    
    configs = [
        {"target_tokens": 10000, "count": 100},
        {"target_tokens": 32000, "count": 100},
        {"target_tokens": 64000, "count": 100},
        {"target_tokens": 128000, "count": 100},
    ]
    
    all_stats_data = {"statistics_per_file": {}}

    print("\nGenerating prompts and writing to individual files...")
    for config in configs:
        target_tokens = config['target_tokens']
        prompt_count = config['count']
        
        print(f"\nCreating {prompt_count} prompts of ~{target_tokens} tokens...")
        
        prompts_for_this_config = []
        token_lengths_for_this_config = []

        for i in tqdm(range(prompt_count)):
            # --- MODIFIED PART ---
            # Now captures both the prompt and the accurate token count directly
            prompt, token_count = create_prompt_of_length(texts, target_tokens, tolerance=0.1)
            # --- END MODIFICATION ---

            token_lengths_for_this_config.append(token_count)
            entry = {"prompt": prompt}
            prompts_for_this_config.append(entry)
            
        output_file = os.path.join(output_dir, f"long_prompts_{target_tokens}_tokens.jsonl")
        print(f"Writing {len(prompts_for_this_config)} prompts to {output_file}")
        
        with open(output_file, 'w') as f:
            for prompt_entry in prompts_for_this_config:
                f.write(json.dumps(prompt_entry) + '\n')
        
        avg_tokens = np.mean(token_lengths_for_this_config)
        min_tokens_val = np.min(token_lengths_for_this_config)
        max_tokens_val = np.max(token_lengths_for_this_config)
        std_tokens = np.std(token_lengths_for_this_config)

        print("\n" + "="*50)
        print(f"Statistics for {output_file}:")
        print("="*50)
        print(f"  - Prompts generated: {len(prompts_for_this_config)}")
        print(f"  - Average tokens:    {avg_tokens:.0f}")
        print(f"  - Minimum tokens:    {min_tokens_val}")
        print(f"  - Maximum tokens:    {max_tokens_val}")
        print(f"  - Std deviation:     {std_tokens:.0f}")
        print("="*50)

        all_stats_data["statistics_per_file"][f"{target_tokens}_tokens"] = {
            "count": prompt_count,
            "output_file": os.path.basename(output_file),
            "average_tokens": float(avg_tokens),
            "min_tokens": int(min_tokens_val),
            "max_tokens": int(max_tokens_val),
            "std_deviation": float(std_tokens),
        }

    stats_file = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(all_stats_data, f, indent=2)
    
    print(f"\nConsolidated statistics for all files saved to {stats_file}")
    print("Dataset generation complete.")

if __name__ == "__main__":
    create_dataset()
