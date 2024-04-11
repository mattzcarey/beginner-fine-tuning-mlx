from mlx_lm.utils import convert


def convert_to_mlx(hf_path: str, mlx_path: str):
    convert(hf_path, mlx_path, True)

    print(f"\nModel converted.\n")
