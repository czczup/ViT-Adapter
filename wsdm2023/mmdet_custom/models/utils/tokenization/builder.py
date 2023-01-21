from .tokenization_clip import MaskClipTokenizer


def build_tokenizer(tokenizer):
    if tokenizer['name']=='clip_tokenizer':
        return MaskClipTokenizer(tokenizer['max_sent_len'])