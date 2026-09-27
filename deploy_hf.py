import sys
import os
from huggingface_hub import HfApi, create_repo

token = sys.argv[1]
api = HfApi(token=token)

try:
    # Get user info
    user_info = api.whoami()
    username = user_info['name']
    print(f"Logged in as {username}")

    repo_id = f"{username}/onion-quality-api"
    
    # Create Space
    print(f"Creating Space: {repo_id}")
    try:
        create_repo(repo_id, token=token, repo_type="space", space_sdk="docker", exist_ok=True)
    except Exception as e:
        print(f"Space might already exist or error: {e}")

    # Upload files
    print("Uploading files...")
    api.upload_folder(
        folder_path="cloud_deployment",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Initial deployment"
    )
    
    print(f"SUCCESS_URL: https://{username}-onion-quality-api.hf.space")

except Exception as e:
    print(f"ERROR: {e}")
