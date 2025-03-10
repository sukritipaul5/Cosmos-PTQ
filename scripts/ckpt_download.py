from huggingface_hub import login, snapshot_download
import os

login(token="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", add_to_git_credential=True)
model_names = [
        "Cosmos-Tokenizer-CI8x8",
        "Cosmos-Tokenizer-CI16x16",
        "Cosmos-Tokenizer-CV4x8x8",
        "Cosmos-Tokenizer-CV8x8x8",
        "Cosmos-Tokenizer-CV8x16x16",
        "Cosmos-Tokenizer-DI8x8",
        "Cosmos-Tokenizer-DI16x16",
        "Cosmos-Tokenizer-DV4x8x8",
        "Cosmos-Tokenizer-DV8x8x8",
        "Cosmos-Tokenizer-DV8x16x16",
]
for model_name in model_names:
    hf_repo = "nvidia/" + model_name
    local_dir = "pretrained_ckpts/" + model_name
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {model_name}...")
    snapshot_download(repo_id=hf_repo, allow_patterns=["*.jit"], local_dir=local_dir)