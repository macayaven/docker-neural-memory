#!/usr/bin/env python3
"""Deploy to HuggingFace Spaces."""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set, skipping deployment")
        return

    api = HfApi()
    repo_id = "macayaven/docker-neural-memory"

    # Create the Space if it doesn't exist
    try:
        create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        print(f"Space {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Create a temporary directory with all required files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Copy deploy/huggingface contents
        shutil.copytree("deploy/huggingface", tmppath / "root", dirs_exist_ok=True)

        # Copy src/ folder (needed for imports)
        shutil.copytree("src", tmppath / "root" / "src", dirs_exist_ok=True)

        # Move files from root to tmpdir
        for item in (tmppath / "root").iterdir():
            shutil.move(str(item), str(tmppath / item.name))
        (tmppath / "root").rmdir()

        print(f"Prepared deployment folder with contents: {list(tmppath.iterdir())}")

        # Upload the combined folder
        api.upload_folder(
            folder_path=str(tmppath),
            repo_id=repo_id,
            repo_type="space",
        )
    print(f"Deployed to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
