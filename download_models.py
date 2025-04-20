import os
import argparse
from huggingface_hub import snapshot_download

MODEL_CHECKSUMS = {
    "TripoSG": {
        "required_files": ["config.json", "pytorch_model.bin"],
        "repo_id": "VAST-AI/TripoSG"
    },
    "RMBG-1.4": {
        "required_files": ["model.onnx", "vocab.txt"],
        "repo_id": "briaai/RMBG-1.4"
    }
}

def check_model_complete(model_dir, required_files):
    if not os.path.exists(model_dir):
        return False
    return all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".", help="Base output directory for models")
    args = parser.parse_args()

    print(f"Checking models in {args.output_dir}...")
    for model_name, config in MODEL_CHECKSUMS.items():
        model_dir = os.path.join(args.output_dir, "pretrained_weights", model_name)
        if check_model_complete(model_dir, config["required_files"]):
            print(f"{model_name} model already exists and is complete at {model_dir}")
        else:
            print(f"Downloading {model_name} model to {model_dir}...")
            os.makedirs(model_dir, exist_ok=True)
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt"]
            )
            print(f"{model_name} download completed")

    print("All models verified and ready.")

if __name__ == "__main__":
    main()