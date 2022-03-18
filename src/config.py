import re
import typing as tp
import os

from pydantic import BaseModel, validator
import pandas as pd
import sentencepiece as sp
import torch


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

    batch_size: int = 3
    epochs: int = 10
    lr: float = 0.001
    gradient_accumulation: int = 8

    embedding_dim: int = 150
    hidden_size: int = 192
    num_layers: int = 2
    dropout: float = 0.5
    bidirectional: bool = True

    dataset_path: str = 'data/conala'

    start_sent: str = '<sos>'
    end_sent: str = '<eos>'
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'
    device: torch.device = torch.device('cuda')

    datasets: tp.Optional[tp.Tuple[pd.DataFrame, pd.DataFrame]] = None
    intent_vocab: tp.Optional[sp.SentencePieceProcessor] = None
    snippet_vocab: tp.Optional[sp.SentencePieceProcessor] = None

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

    @validator('intent_vocab', always=True)
    def init_intent_vocab(cls, v, values):
        return sp.SentencePieceProcessor(model_file='data/for_bpe/intent.model')

    @validator('snippet_vocab', always=True)
    def init_snippet_vocab(cls, v, values):
        return sp.SentencePieceProcessor(model_file='data/for_bpe/snippet.model')

