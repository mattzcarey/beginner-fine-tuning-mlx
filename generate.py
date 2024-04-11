import argparse

from mlx_lm import generate, load

from constants import adapter_path, hf_path


def test_generate(prompt, path=hf_path, adapters=adapter_path):
    model, tokenizer = load(path, adapter_path=adapters)
    generate(model, tokenizer, prompt=prompt, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with and without trained adapters."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt the model.",
    )
    args = parser.parse_args()

    test_generate(args.prompt)
