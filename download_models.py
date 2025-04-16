import os
from huggingface_hub import snapshot_download

print("Downloading TripoSG model...")
triposg_weights_dir = "pretrained_weights/TripoSG"
if not os.path.exists(triposg_weights_dir):
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir, local_dir_use_symlinks=False)
else:
    print("TripoSG model already downloaded.")

print("Downloading RMBG model...")
rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
if not os.path.exists(rmbg_weights_dir):
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir, local_dir_use_symlinks=False)
else:
    print("RMBG model already downloaded.")

print("Model downloads complete.") 