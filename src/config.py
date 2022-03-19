import re
import typing as tp
import os

from pydantic import BaseModel, validator
import pandas as pd
import sentencepiece as sp
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class Config(BaseModel):
    seed: int = 25

    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-2
    gradient_accumulation: int = 4

    scheduler: tp.Any = CosineAnnealingLR
    scheduler_kwargs: tp.Mapping = {
        'T_max': 5,
        'eta_min': 1e-4,
    }

    embedding_dim: int = 300
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.5
    bidirectional: bool = True
    label_smoothing: float = 0.0
    teacher_forcing_ratio: float = 0.6

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
        dirty_df = pd.read_csv('data/homework_data/train.csv')
        dirty_df = dirty_df[
            dirty_df['intent'].str.contains('[P|p]ython', regex=True) &
            (dirty_df['snippet'].apply(len) < 80)
        ]

        train_df = pd.concat(
            [
                pd.read_csv(os.path.join(values['dataset_path'], 'train.csv')),
                pd.read_csv(os.path.join(values['dataset_path'], 'valid.csv')),
                dirty_df[:2000],
            ],
            ignore_index=True,
        )
        valid_df = pd.read_csv(os.path.join(values['dataset_path'], 'test.csv'))
        return train_df, valid_df
