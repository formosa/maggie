from pathlib import Path
import os
from huggingface_hub import snapshot_download


def download_model(repo_user="", repo_name="", target_dir=None, parent_dir=None, token_path="_hf_access.token", force_download=False, **kwargs) -> str:
    """
    
    Downloads a model from Hugging Face Hub and saves it to a specified directory.

    If the directory does not exist, it will be created.
    If no directory is specified, the model will be downloaded to the current working directory.

    Args:
        repo_user (str): The username of the model repository owner.
        repo_name (str): The name of the model repository.
        target_dir (str, optional): The directory where the model will be downloaded. Defaults to None.
        parent_dir (str, optional): The parent directory where the model will be downloaded. Defaults to None.
        token_path (str, optional): Path to the file containing the Hugging Face token. Defaults to "_hf_access.token".
        force_download (bool, optional): If True, forces the download even if the model already exists. Defaults to False.
        **kwargs: Additional arguments for snapshot_download.

    Returns:
        str: The path to the downloaded model.

    Raises:
        Exception: If an error occurs during the download process.
    """

    local_dir = (
        Path(f"{parent_dir}/{repo_user}/{repo_name}") if parent_dir
        else Path(target_dir) if target_dir
        else None
    )

    try:
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"Created directory: {local_dir}")

        return snapshot_download(
            repo_id=f"{repo_user}/{repo_name}", 
            local_dir=local_dir, 
            token=Path(token_path).read_text().strip(),
            force_download=force_download,
            **kwargs
        )

    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
        raise
    

if __name__ == "__main__":

    repo_user = "neuralmagic"
    repo_name = "Mistral-7B-Instruct-v0.3-GPTQ-4bit"

    parent_path = "C:/AI/claude/service/maggie/maggie/models"
    token_path = r"C:\AI\claude\service\maggie\_hf_access.token"
    
    try:
        print(
            download_model(
            repo_user=repo_user, 
            repo_name=repo_name, 
            parent_dir=parent_path, 
            token_path=token_path
            )
        )

    except Exception as e:
        print(f"An error occurred: {e}")