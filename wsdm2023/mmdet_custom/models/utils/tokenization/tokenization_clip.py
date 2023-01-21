import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import torch
from torch._C import Value
import pandas as pd


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"),
                                                      ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class ClipTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>',
                     '<|mask|>', '<|gen|>', '<|spe|>'])  # vocablength=49411
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.encoder = pd.Series(
            list(self.encoder.values()), index=self.encoder.keys())

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.decoder = pd.Series(
            list(self.decoder.values()), index=self.decoder.keys())

        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>',
                      '<|mask|>': '<|mask|>', '<|gen|>': '<|gen|>', '<|spe|>': '<|spe|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|<\|gen\|>|<\|spe\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

        self.vocab = self.encoder
        self.ids_to_tokens = self.decoder

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors="replace").replace('</w>', ' ')
        return text

    def basic_tokenize(self, text):
        text = whitespace_clean(basic_clean(text)).lower()
        return list(re.findall(self.pat, text))

    def encode_basic_tokenized_token(self, token):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        bpe_tokens = [self.encoder[bpe_token]
                      for bpe_token in self.bpe(token).split(' ')]
        return bpe_tokens

    def tokenize(self, text):
        tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder[bpe_token] for bpe_token in tokens]

    def add_special_tokens_single_sentence(self, token_ids, start_type='SoT'):
        if start_type == 'SoT':
            return [self.encoder['<|startoftext|>']] + token_ids + [self.encoder['<|endoftext|>']]
        elif start_type == 'Gen':
            return [self.encoder['<|gen|>']] + token_ids + [self.encoder['<|endoftext|>']]
        elif start_type == 'SPE':
            return token_ids
        else:
            raise ValueError

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1, start_type='SoT'):
        sep = [self.encoder['<|endoftext|>']]
        if start_type == 'SoT':
            cls = [self.encoder['<|startoftext|>']]
        elif start_type == 'Gen':
            cls = [self.encoder['<|gen|>']]
        elif start_type == 'SPE':
            cls = []
        else:
            raise ValueError
        return cls + token_ids_0 + sep + token_ids_1

    def get_cls_token_id(self):
        return self.encoder['<|startoftext|>']

    def get_eos_token_id(self):
        return self.encoder['<|endoftext|>']

    def convert_tokens_to_string(self, tokens):
        text = ''.join(tokens).strip()
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors="replace").replace('</w>', ' ')
        return text


class MaskClipTokenizer(ClipTokenizer):
    def __init__(self, max_sent_len=128, bpe_path: str = default_bpe()):
        super().__init__(bpe_path)
        self.max_sent_len = max_sent_len

    def encode(self, text):
        input_ids = torch.tensor(super().encode(text))
        
        if len(input_ids) > self.max_sent_len:
            input_ids = input_ids[0:self.max_sent_len]
            mask = torch.ones_like(input_ids)
        elif len(input_ids) < self.max_sent_len:
            mask = torch.ones_like(input_ids)
            pad = torch.zeros(
                [self.max_sent_len-len(input_ids)], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=0)
            mask = torch.cat([mask, pad], dim=0)
        
        return input_ids,mask
