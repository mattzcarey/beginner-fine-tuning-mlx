import json

import sentencepiece as spm

"""The code below counts the number of characters in your training dataset to figure out how long training will take. It's not high accuracy."""


def count_tokens(
    mlx_path,
    train_set,
    epochs,
    lora_layers,
):
    model_path = f"{mlx_path}/tokenizer.model"
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.load(model_path)

    def count_tokens_in_file(sp_processor, file_path):
        total_tokens = 0
        with open(file_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get(
                    "text", ""
                )  # Make sure 'text' is the correct key for your JSONL entries
                tokens = sp_processor.encode_as_pieces(text)
                total_tokens += len(tokens)
        return total_tokens

    # Use the correct variable for the SentencePiece processor here
    total_tokens = (
        count_tokens_in_file(sp_processor, f"{train_set}/train.jsonl") * epochs
    )
    multiplier_for_layers = 1 / (
        1 + (((32 - lora_layers) / 32) * 1.5)
    )  # Really approximate maths to get a multiplier based on number of layers.
    training_rate = 500 // multiplier_for_layers
    estimated_total_time = int(total_tokens // training_rate)
    estimated_minutes = int(estimated_total_time // 60)
    estimated_seconds = int(estimated_total_time % 60)
    slow_time = estimated_total_time * 7
    slow_minutes = int(slow_time // 60)
    slow_seconds = int(slow_time % 60)

    print(f"Total number of tokens in the JSONL file: {total_tokens}")
    print(f"Estimated training rate in tokens/second if fits in GPU: {training_rate}")
    print(
        f"\nIf model fits in GPU: Estimated time for {epochs} epoch(s) with {lora_layers} LoRA layer(s) with a token amount of {total_tokens}: \n{estimated_minutes} minutes and {estimated_seconds} seconds"
    )
    print(
        f"\nElse if model doesn't fit in GPU, could be up to:\n{slow_minutes} minutes and {slow_seconds} seconds\n"
    )
