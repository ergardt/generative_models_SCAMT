import torch
from scripts.utils import LstmSeq2SeqEncoder, TransformerModel
from torch import nn
from torch.distributions import Categorical
from torch.nn.modules.activation import Sigmoid
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Generator(nn.Module):

    def __init__(self, latent_dim, vocab_size, start_token, end_token):
        """Generator

        Args:
            latent_dim (int): [description]
            vocab_size (int): vocab size without padding
            start_token ([int]): start token (without padding idx)
            end_token ([int]): end token (without padding idx)
        """

        super().__init__()

        # (-1) we do not need pad token for the generator
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.latent_dim = latent_dim

        self.embedding_layer = nn.Embedding(self.vocab_size, latent_dim)

        # self.project = FeedForward(
        #     input_dim=latent_dim,
        #     num_layers=2,
        #     hidden_dims=[latent_dim * 2, latent_dim * 2],
        #     activations=[nn.ReLU(), nn.ELU(alpha=0.1)],
        #     dropout=[0.1, 0.1]
        # )
        self.project = nn.Sequential(
            nn.Linear(latent_dim,latent_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim*2, latent_dim * 2),
            nn.ELU(alpha=0.1),
            nn.Dropout(0.1),
        )
        # self.rnn = nn.LSTMCell(latent_dim, latent_dim)
        self.rnn = nn.LSTM(latent_dim, latent_dim, num_layers=2, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, vocab_size)
        )

    def forward(self, z, max_len=100):
        """[summary]

        Args:
            z (torch.Tensor): [description]
            max_len (int, optional): [description]. Defaults to 20.

        Returns:
            dict: x [B, max_len], log_probabilities [B, max_len, vocab], entropies [B,]
        """

        batch_size = z.shape[0]

        # start of sequence
        starts = torch.full(
            size=(batch_size,), fill_value=self.start_token, device=z.device).long()

        # embed_start
        emb = self.embedding_layer(starts)

        x = []
        log_probabilities = []
        entropies = []

        # h, c = self.project(z).chunk(2, dim=1)

        hc = self.project(z)
        h0 = torch.zeros(2, batch_size, self.latent_dim).to(z.device)
        c0 = torch.zeros(2, batch_size, self.latent_dim).to(z.device)
        h0[0] = hc[:, :self.latent_dim]
        c0[0] = hc[:, self.latent_dim:]
        current_token = torch.full((batch_size,), self.start_token, device=z.device).long()
        h, c = h0, c0



        for i in range(max_len):

            emb = self.embedding_layer(current_token).unsqueeze(1)  # [B, 1, latent_dim]
            output, (h, c) = self.rnn(emb, (h, c))

            # new state
            # h, c = self.rnn(emb, (h, c))

            # prediction
            logits = self.output_layer(output.squeeze(1))

            # 🔥 ЗАПРЕТИТЬ НЕДОПУСТИМЫЕ ТОКЕНЫ 🔥
            logits[:, 0] = -1e9  # <pad>
            logits[:, self.start_token] = -1e9  # <sos>
            if i < 3:
                logits[:, self.end_token] = -1e9

            # create dist
            dist = Categorical(logits=logits)

            # sample
            sample = dist.sample()

            # append prediction
            x.append(sample)

            # append log prob
            log_probabilities.append(dist.log_prob(sample))

            # append entropy
            entropies.append(dist.entropy())

            # new embedding
            # emb = self.embedding_layer(sample)
            current_token = sample

        # stack along sequence dim
        x = torch.stack(x, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        entropies = torch.stack(entropies, dim=1)

        # keep only valid lengths (before EOS)
        end_pos = (x == self.end_token).float().argmax(dim=1).cpu()

        # sequence length is end token position + 1
        seq_lengths = end_pos + 1

        # if end_pos = 0 => put seq_length = max_len
        seq_lengths.masked_fill_(seq_lengths == 1, max_len)

        # select up to length
        _x = []
        _log_probabilities = []
        _entropies = []
        for x_i, logp, ent, length in zip(x, log_probabilities, entropies, seq_lengths):
            _x.append(x_i[:length])
            _log_probabilities.append(logp[:length])
            _entropies.append(ent[:length].mean())

        x = torch.nn.utils.rnn.pad_sequence(
            _x, batch_first=True, padding_value=0)

        # x = x + 1  # add padding token

        return {'x': x, 'log_probabilities': _log_probabilities, 'entropies': _entropies}




# В scripts.layers — замените RecurrentDiscriminator на:

class RecurrentDiscriminator(nn.Module):
    def __init__(self, hidden_size, vocab_size, start_token, bidirectional=True):
        super().__init__()
        self.start_token = start_token
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.rnn = LstmSeq2SeqEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=bidirectional,
            dropout=0.1
        )

        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        self.critic_head = nn.Sequential(
            nn.Linear(rnn_output_size, rnn_output_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(rnn_output_size, 1)  # ← нет Sigmoid!
        )

# В RecurrentDiscriminator (layers.py)
    def forward(self, x, from_embeddings=False):
        if from_embeddings:
            # x — это уже эмбеддинги: [B, L, H]
            emb = x
            batch_size, seq_len, _ = emb.shape
            
            # Маска: предполагаем, что padding — нулевые эмбеддинги
            # Но лучше передавать маску отдельно! Пока сделаем приближение:
            mask = (emb.abs().sum(dim=-1) != 0)  # [B, L]
            
            # Добавляем <sos> эмбеддинг в начало
            sos_emb = self.embedding(torch.tensor(self.start_token, device=emb.device)).unsqueeze(0).unsqueeze(0)  # [1,1,H]
            sos_emb = sos_emb.expand(batch_size, 1, -1)
            emb = torch.cat([sos_emb, emb], dim=1)  # [B, L+1, H]
            mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=emb.device), mask], dim=1)
        else:
            # Обычный режим: x — токены (LongTensor)
            batch_size, seq_len = x.shape
            starts = torch.full((batch_size, 1), self.start_token, device=x.device, dtype=torch.long)
            x = torch.cat([starts, x], dim=1)  # [B, L+1]
            mask = (x != 0)
            emb = self.embedding(x)  # [B, L+1, H]
        
        rnn_out = self.rnn(emb, mask)
        masked_out = rnn_out * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        seq_repr = masked_out.sum(dim=1) / lengths
        return self.critic_head(seq_repr)
    


import torch
import torch.nn as nn
from scripts.utils import TransformerModel  # убедитесь, что он возвращает [B, L, H]


from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

class TransformerDiscriminator(nn.Module):
    def __init__(self, hidden_size, vocab_size, max_len, start_token, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.start_token = start_token
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.embed_layer_norm = LayerNorm(hidden_size)  # ← ДОБАВЛЕНО
        
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len + 1, hidden_size))  # +1 для <sos>


        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        nn.init.normal_(self.critic_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.critic_head[-1].bias, 0.0)

    def forward(self, x, from_embeddings=False):
        if from_embeddings:
            emb = x
            batch_size, seq_len, _ = emb.shape
            mask = (emb.abs().sum(dim=-1) != 0)

            # Добавляем <sos> эмбеддинг
            sos_emb = self.embedding(torch.tensor(self.start_token, device=emb.device)).unsqueeze(0).unsqueeze(0)
            sos_emb = sos_emb.expand(batch_size, 1, -1)
            emb = torch.cat([sos_emb, emb], dim=1)
            mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=emb.device), mask], dim=1)
            
            # Применяем LayerNorm после конкатенации (включая <sos>)
            emb = self.embed_layer_norm(emb)
            
        else:
            batch_size, seq_len = x.shape
            starts = torch.full((batch_size, 1), self.start_token, device=x.device, dtype=torch.long)
            x = torch.cat([starts, x], dim=1)
            x = x[:, :self.max_len + 1]
            mask = (x != 0)
            emb = self.embedding(x)
            
            # Применяем LayerNorm ко всей последовательности (включая <sos>)
            emb = self.embed_layer_norm(emb)
        
        # Позиционное кодирование
        emb = emb + self.pos_encoding[:, :emb.size(1), :]
        
        # Transformer
        transformer_out = self.transformer(emb, src_key_padding_mask=~mask)
        
        # Глобальный пуллинг
        masked_out = transformer_out * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        seq_repr = masked_out.sum(dim=1) / lengths
        
        return self.critic_head(seq_repr)