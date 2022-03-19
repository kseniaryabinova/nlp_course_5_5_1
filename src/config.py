import re
import typing as tp
import os

from pydantic import BaseModel, validator
import pandas as pd
import sentencepiece as sp
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def _preprocess(dataset_filepath: str, column: str):
    df = pd.concat([
        pd.read_csv(os.path.join(dataset_filepath, 'train.csv')),
        pd.read_csv(os.path.join(dataset_filepath, 'valid.csv')),
        pd.read_csv(os.path.join(dataset_filepath, 'test.csv')),
    ])
    df[column] = df[column].str.lower()
    if column == 'intent':
        df[column] = df[column].apply(lambda x: list(re.findall(r"[\w']+", x)))
    elif column == 'snippet':
        df[column] = df[column].apply(lambda x: list(
            re.findall(r"[\w']+|[.,!?;:@~(){}\[\]+-/=\\\'\"\`]", x),
        ))
    return df


class Config(BaseModel):
    seed: int = 25

    batch_size: int = 24
    epochs: int = 40
    lr: float = 1e-3
    gradient_accumulation: int = 3

    scheduler: tp.Any = CosineAnnealingLR
    scheduler_kwargs: tp.Mapping = {
        'T_max': 10,
        'eta_min': 1e-5,
    }

    embedding_dim: int = 150
    hidden_size: int = 192
    num_layers: int = 4
    dropout: float = 0.5
    bidirectional: bool = True
    label_smoothing: float = 0.0
    teacher_forcing_ratio: float = 0.5

    dataset_path: str = 'data/conala'

    start_sent: str = '<s>'
    end_sent: str = '</s>'
    device: torch.device = torch.device('cuda')

    datasets: tp.Optional[tp.Tuple[pd.DataFrame, pd.DataFrame]] = None
    intent_vocab: tp.Optional[sp.SentencePieceProcessor] = sp.SentencePieceProcessor(model_file='data/for_bpe/intent.model')
    snippet_vocab: tp.Optional[sp.SentencePieceProcessor] = sp.SentencePieceProcessor(model_file='data/for_bpe/snippet.model')

    class Config:
        arbitrary_types_allowed = True

    @validator('datasets', always=True)
    def init_dataset(cls, v, values):
        train_df = pd.concat([
            pd.read_csv(os.path.join(values['dataset_path'], 'train.csv')),
            pd.read_csv(os.path.join(values['dataset_path'], 'valid.csv')),
        ])
        valid_df = pd.read_csv(os.path.join(values['dataset_path'], 'test.csv'))
        return train_df, valid_df
