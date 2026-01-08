#!/usr/bin/env python3
"""Deploy to HuggingFace Spaces."""

import os
import sys

from huggingface_hub import HfApi, create_repo


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set, skipping deployment")
        return

    api = HfApi()
    repo_id = "macayaven/docker-neural-memory"

    # Create the Space if it doesn't exist
    try:
        create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        print(f"Space {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the deployment folder
    api.upload_folder(
        folder_path="deploy/huggingface",
        repo_id=repo_id,
        repo_type="space"
    )
    print(f"Deployed to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
