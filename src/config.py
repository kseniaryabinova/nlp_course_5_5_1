import re
import typing as tp
import os

from pydantic import BaseModel, validator
import pandas as pd
from torchtext.vocab import Vocab, build_vocab_from_iterator


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

    batch_size: int = 2
    epochs: int = 10
    lr: float = 0.001

    embedding_dim: int = 12
    hidden_size: int = 10
    num_layers: int = 1
    dropout: float = 0.8
    bidirectional: bool = False

    dataset_path: str = 'data/homework_data'

    start_sent: str = '<sos>'
    end_sent: str = '<eos>'
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'
    device: str = 'cpu'

    datasets: tp.Optional[tp.Tuple[pd.DataFrame, pd.DataFrame]] = None
    intent_vocab: tp.Optional[Vocab] = None
    snippet_vocab: tp.Optional[Vocab] = None

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
        df = _preprocess(values['dataset_path'], 'intent')
        vocab: Vocab = build_vocab_from_iterator(
            iter(df['intent']),
            specials=[
                values['pad_token'],
                values['unk_token'],
                values['start_sent'],
                values['end_sent'],
            ]
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    @validator('snippet_vocab', always=True)
    def init_snippet_vocab(cls, v, values):
        df = _preprocess(values['dataset_path'], 'snippet')
        vocab: Vocab = build_vocab_from_iterator(
            iter(df['snippet']),
            specials=[
                values['pad_token'],
                values['unk_token'],
                values['start_sent'],
                values['end_sent'],
            ]
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab
