""" Example usage: python build_training_data.py all_too_well_lyrics.txt creates all_too_well_lyrics.pt """
import os
from typing import List

import fire
import torch

PAD_TOKEN = "__PAD__"
END_TOKEN = "__END__"


def make_word_level_tokenizer(text):
    tokenizer = {word: idx + 2 for idx, word in enumerate(set(text))}
    tokenizer[PAD_TOKEN] = 0
    tokenizer[END_TOKEN] = 1

    return tokenizer


def turn_text_into_repeated_tokens(
    text_path: str, desired_token_len=512000
) -> List[str]:
    with open(text_path) as f:
        flattened_text = f.read().split()

    tokenizer = make_word_level_tokenizer(flattened_text)

    tokenized = [tokenizer[w] for w in flattened_text + [END_TOKEN]]
    token_tensor = torch.tensor(tokenized * (desired_token_len // len(tokenized)))

    token_path = os.path.splitext(text_path)[0] + ".pt"
    torch.save(token_tensor, token_path)


if __name__ == "__main__":
    fire.Fire(turn_text_into_repeated_tokens)
