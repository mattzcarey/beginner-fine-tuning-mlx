import json
import os


def convert_md_to_jsonl(md_file_path):
    with open(md_file_path, "r") as md_file:
        content = md_file.read()

    # Remove the markdown header
    header_end = content.find("---", content.find("---") + 3)
    if header_end != -1:
        content = content[header_end + 3 :]

    lines = content.lstrip().split("\n\n")

    txt_data = [line for line in lines if line and not line.startswith(("|", ">"))]

    return txt_data


def build_dataset(content_dir, data_dir, push_to_hub):
    data = []
    for filename in os.listdir(content_dir):
        if filename.endswith(".md"):
            md_file_path = os.path.join(content_dir, filename)
            data.extend(convert_md_to_jsonl(md_file_path))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(f"{data_dir}/dataset.txt", "w") as test_dataset_file:
        for item in data:
            test_dataset_file.write(json.dumps(item) + "\n")

    if push_to_hub:
        from datasets import load_dataset

        dataset = load_dataset(
            "text",
            data_files={
                "train": f"{data_dir}/dataset.txt",
            },
        )

        dataset.push_to_hub("mattzcarey/ready-set-cloud-blogs", private=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert .md files in a directory to .jsonl format."
    )
    parser.add_argument(
        "--content_dir",
        type=str,
        help="The directory of .md files to convert.",
        default="./blogs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The directory to save the converted .jsonl file.",
        default="./data",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the dataset to the Hub.",
        default=False,
    )

    args = parser.parse_args()
    build_dataset(args.content_dir, args.data_dir, args.push_to_hub)
