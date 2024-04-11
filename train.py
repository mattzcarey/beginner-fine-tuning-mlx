import subprocess

from constants import (
    adapter_path,
    batch_size,
    context_length,
    epochs,
    hf_path,
    learning_rate,
    lora_layers,
    mlx_path,
    train_file,
    train_set,
)
from convert import convert_to_mlx
from dataset.format import format_dataset
from utils.count_iters import count_iters
from utils.count_tokens import count_tokens

# convert model to mlx
convert_to_mlx(hf_path, mlx_path)

# build training file
format_dataset(hf_path, train_set)

iters = count_iters(batch_size, train_set, epochs)
count_tokens(mlx_path, train_set, epochs, lora_layers)

args = [
    "python",
    "-m",
    "mlx_lm.lora",
    "--model",
    mlx_path,
    "--train",
    "--data",
    train_set,
    "--iters",
    str(iters),
    "--max-seq-length",
    str(context_length),
    "--batch-size",
    str(batch_size),
    "--learning-rate",
    str(learning_rate),
    "--lora-layers",
    str(lora_layers),
    "--adapter-path",
    adapter_path,
    # "--resume-adapter-file",
    # adapter_path
]

subprocess.run(args)

print(f"\nModel training complete.\n")
