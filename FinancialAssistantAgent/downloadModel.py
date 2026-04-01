from huggingface_hub import hf_hub_download
from pathlib import Path

# Target folder to store the model
target_dir = Path("C:/Users/222003150/project/FinancialAssistantAI/models")
target_dir.mkdir(parents=True, exist_ok=True)

# Download model from Hugging Face
file_path = hf_hub_download(
    repo_id="QuantFactory/finance-Llama3-8B-GGUF",
    filename="finance-Llama3-8B.Q2_K.gguf",  # make sure this matches the repo file
    cache_dir=str(target_dir)  # save in your models folder
)

print("Downloaded to:", file_path)

