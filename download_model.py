from huggingface_hub import snapshot_download

print("Downloading Poetry-Expert locally...")
math_path = snapshot_download(
    repo_id='kzykazzam/qwen-poetry-finetuned',
    cache_dir='local_models',
    local_dir='local_models/poetry-expert',
    local_dir_use_symlinks=False
)
print(f"Math-Expert: {math_path}")