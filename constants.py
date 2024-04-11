# model
hf_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
mlx_path = f"models/{hf_path}-mlx"
adapter_path = f"{mlx_path}/adapters"

# hyperparameters
batch_size = 1
epochs = 1
train_set = "./data"  # Path to train.jsonl and valid.jsonl
lora_layers = 22
context_length = 1024
learning_rate = 2e-5
