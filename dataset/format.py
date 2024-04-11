import json

from datasets import load_dataset
from transformers import AutoTokenizer


def write_dataset_to_file(dataset, file_path, tokenizer):
    EOS_TOKEN = tokenizer.eos_token
    with open(file_path, "w") as file:
        for item in dataset:
            story_with_eos = (
                item["text"] + EOS_TOKEN
            )  # Append the EOS token to each story
            # Write this story as a JSON object to the file
            file.write(json.dumps({"text": story_with_eos}) + "\n")


def format_dataset(model, data_dir, validation_split=0.1):
    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset = load_dataset(
        "text",
        data_files={
            "train": f"{data_dir}/dataset.txt",
        },
    )

    # Split the dataset into training and validation sets
    dataset = dataset["train"].train_test_split(test_size=validation_split)

    # Process the training set
    write_dataset_to_file(dataset["train"], f"{data_dir}/train.jsonl", tokenizer)

    # Process the validation set
    write_dataset_to_file(dataset["test"], f"{data_dir}/valid.jsonl", tokenizer)

    print(f"\nEOS token added to each entry.\n")
