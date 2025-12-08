import torch


class Tokenizer(object):

    def __init__(self, data, max_len=100):

        unique_char = list(set(''.join(data))) + ['<eos>'] + ['<sos>']

        self.mapping = {'<pad>': 0}

        for i, c in enumerate(unique_char, start=1):
            self.mapping[c] = i

        self.inv_mapping = {v: k for k, v in self.mapping.items()}

        self.start_token = self.mapping['<sos>']

        self.end_token = self.mapping['<eos>']

        self.vocab_size = len(self.mapping.keys())
        self.max_len = max_len

    def encode_smile(self, mol, add_eos=True):

        out = [self.mapping[i] for i in mol]

        if add_eos:
            out = out + [self.end_token]

        return torch.LongTensor(out)

    def batch_tokenize(self, batch: list[str]) -> torch.LongTensor:
        sequences = []
        for smiles in batch:
            # Токенизация одного SMILES (без <sos>, но с <eos>, как у вас)
            tokens = self.tokenize(smiles)  # должен возвращать список id
            # Обрежем, если длиннее
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            else:
                # Дополним нулями (padding_idx=0) до max_len
                tokens += [0] * (self.max_len - len(tokens))
            sequences.append(tokens)
        return torch.LongTensor(sequences)
        

    # В класс Tokenizer
    def tokenize(self, smiles: str) -> list[int]:
        """Convert a single SMILES string to token ids."""
        # Ваша логика токенизации (обычно: разбить на символы или подсловы)
        # Пример для символьного токенизатора:
        tokens = list(smiles)
        ids = []
        for t in tokens:
            if t in self.mapping:
                ids.append(self.mapping[t])
            else:
                ids.append(self.mapping['<unk>'])  # или выбросить ошибку
        # Добавить <eos>, если нужно
        ids.append(self.end_token)
        return ids