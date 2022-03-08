import re

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import Config
from src.const import TARGETS, INPUTS, LENGTHS


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df.copy()
        self.intent_vocab = config.intent_vocab
        self.snippet_vocab = config.snippet_vocab
        self.sequence_start = self.intent_vocab([config.start_sent])
        self.sequence_end = self.intent_vocab([config.end_sent])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        intent = self.df.iloc[index, 0]
        snippet = self.df.iloc[index, 1]

        intent_token_ids = self.intent_vocab(tokenize_question(intent))
        snippet_token_ids = self.snippet_vocab(tokenize_snippet(snippet))

        x = self.sequence_start + intent_token_ids + self.sequence_end
        y = snippet_token_ids + self.sequence_end
        return x, len(x), y


def tokenize_question(text):
    """
    Tokenizes question from a string into a list of strings (tokens)
    """
    text = text.lower()
    return list(
        filter(lambda x: len(x) < 16, re.findall(r"[\w']+", text)),
    )


def tokenize_snippet(text):
    """
    Tokenizes code snippet into a list of operands.
    """
    text = text.lower()
    return list(filter(
        lambda x: len(x) < 10,
        re.findall(r"[\w']+|[.,!?;:@~(){}\[\]+-/=\\\'\"\`]", text),
    ))


def get_datasets(config: Config):
    train_dataset = TextDataset(config.datasets[0], config)
    valid_dataset = TextDataset(config.datasets[1], config)
    return train_dataset, valid_dataset


def get_dataloaders(config: Config):
    train_dataset, valid_dataset = get_datasets(config)
    pad_token_id = config.intent_vocab([config.pad_token])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_token_id),
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_token_id),
    )

    return train_dataloader, valid_dataloader


def collate_fn(batch, pad_token_id):
    intent_max_len = max([len(x) for x, _, _ in batch])
    snippet_max_len = max([len(x) for _, _, x in batch])
    padded_x = []
    padded_y = []
    input_lens = []
    for x, lens, y in batch:
        padded_x.append(x + pad_token_id * (intent_max_len - len(x)))
        padded_y.append(y + pad_token_id * (snippet_max_len - len(y)))
        input_lens.append(lens)

    return {
        INPUTS: (
            torch.tensor(padded_x).T,
            torch.tensor(input_lens),
            torch.tensor(padded_y).T,
        ),
        TARGETS: torch.tensor(padded_y).T,
    }
