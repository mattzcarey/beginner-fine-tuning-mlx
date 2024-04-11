import argparse

from mlx_lm import generate, load

from constants import adapter_path, hf_path


def test_generate(path, adapters):
    prompt = "## The next step on the Serverless journey\n"

    print(f"\n**Without your trained adapters:**")
    model, tokenizer = load(path)
    response = generate(model, tokenizer, prompt=prompt, verbose=True)

    print(f"\n**And now with your trained adapters:**")
    model, tokenizer = load(path, adapter_path=adapters)
    response = generate(model, tokenizer, prompt=prompt, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with and without trained adapters."
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        help="The repo id for the hf model.",
        default=hf_path,
    )
    parser.add_argument(
        "--adapters",
        type=str,
        help="The path to the trained adapters.",
        default=adapter_path,
    )
    args = parser.parse_args()

    test_generate(args.hf_path, args.adapters)
