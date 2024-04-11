# Beginner's guide to fine-tuning with MLX (Mac only)

This repo is a beginner's guide to creating a fine-tuned text completions model. The aim is to make is as simple as possible to generate new content in the same style and tone as existing content.

## Setup

Setup a virtual environment. Use any virtual environment manager.
```bash
pyenv install 3.11.7
pyenv virtualenv 3.11.7 mlx
pyenv activate mlx
```

Install the required packages
```bash
poetry install --no-root
```

## Usage

Add content such as blogs or articles in md format to a `blogs` directory.

Run the following command to create a dataset from the content. You can optionaly push this to HF Hub with `--push_to_hub` flag.
```bash
python dataset/build.py
```

Modify the model or hyperparameters in `constants.py`.

Run the following command to fine-tune the model on the dataset.
```bash
python train.py
```

This might take a while depending on the size of the dataset and the model. You will get an estimate of the time it will take to train the model on a Mac M2 Max. If you have a Pro it will be 2x slower, a standard will be 4x of more slower. An Ultra will be 2x faster.

## Evaluation

Run the following command for the vibes check.
```bash
python test.py
```

You can modify the prompt in `test.py` to see how the model performs on different prompts.

## Generate

Run the following command to generate text from the model.
```bash
python generate.py --prompt "Once upon a time"
```

Credits to:

- [mlx-examples](https://github.com/ml-explore/mlx-examples)
- [Mark Lord's very helpful notebook](https://github.com/mark-lord/MLX-text-completion-notebook)
- [unsloth](https://github.com/unslothai/unsloth)

