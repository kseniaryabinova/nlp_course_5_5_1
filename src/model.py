import typing as tp
import random

import torch
from torch import Tensor
from torch import nn

from src.config import Config


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        dropout,
        config: Config,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(
            emb_dim,
            enc_hid_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
        )

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src sent len, batch size]
        # src_len = [src sent len]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_len.to(torch.device('cpu')),
            # batch_first=True,
            enforce_sorted=False,
        )

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        # mask = [batch size, src sent len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention = [batch size, src sent len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return torch.nn.Softmax(dim=1)(attention)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,
                 attention, config: Config):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(
            (enc_hid_dim * 2) + emb_dim,
            dec_hid_dim,
            # bidirectional=config.bidirectional,
        )

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim,
                             output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        # mask = [batch size, src sent len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src sent len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src sent len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.device = config.device

        attn = Attention(config.hidden_size, config.hidden_size)
        self.encoder = Encoder(
            len(config.intent_vocab),
            config.embedding_dim,
            config.hidden_size,
            config.hidden_size,
            config.dropout,
            config,
        )
        self.decoder = Decoder(
            len(config.snippet_vocab),
            config.embedding_dim,
            config.hidden_size,
            config.hidden_size,
            config.dropout,
            attn,
            config,
        )

        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.device = torch.device(config.device)
        self.teacher_forcing_ratio = config.teacher_forcing_ratio

    def create_mask(self, src):
        mask = src != self.pad_idx
        return mask.T

    def forward(
        self,
        data: tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        src, src_len, trg = data
        src = src.to(self.device)
        src_len = src_len.to(self.device)
        trg = trg.to(self.device)

        # src = [src sent len, batch size]
        # src_len = [batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        if self.teacher_forcing_ratio == 0:
            assert self.teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
            trg = torch.zeros((100, src.shape[1])).long().fill_(
                self.sos_idx).to(src.device)
        else:
            inference = False

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(
            self.device)

        # tensor to store attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(
            self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        output = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src sent len]

        for t in range(1, max_len):
            output, hidden, attention = self.decoder(
                output,
                hidden,
                encoder_outputs,
                mask,
            )
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
            if inference and output.item() == self.eos_idx:
                return outputs[:t]

        return outputs
