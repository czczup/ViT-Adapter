from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from mmdet_custom.models.utils.tokenization import ClipTokenizer
import torch
import json
import numpy as np


@PIPELINES.register_module()
class RandomFlipWithRefer(RandomFlip):
    # only allow horizontal flip
    def __init__(self, flip_ratio=None):
        super().__init__(flip_ratio, 'horizontal')

    def __call__(self, results):
        results = super().__call__(results)
        # if flipped, the direction related word in refer should be reversed
        if results['flip']:
            refer = results['refer']
            refer = refer.replace(
                'right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')
            results['refer'] = refer

        return results


@PIPELINES.register_module()
class LoadRefer:
    def __init__(self, tag='refer') -> None:
        self.tag = tag

    def __call__(self, results):
        info = results['img_info']
        refer = info[self.tag]
        refer = refer.replace('\"', '').replace('?', '').strip(' ').lower()
        results['refer'] = refer
        return results


@PIPELINES.register_module()
class TokenizeRefer:
    def __init__(self, max_sent_len) -> None:
        self.max_sent_len = max_sent_len
        self.tokenizer = ClipTokenizer()

    def __call__(self, results):
        refer = results['refer']
        input_ids = torch.tensor(
            self.tokenizer.encode(refer))
        if len(input_ids) > self.max_sent_len:
            print(f"len(input_ids) > self.max_sent_len! len(input_ids) = {len(input_ids)}")
            input_ids = input_ids[0:self.max_sent_len]
            mask = torch.ones_like(input_ids)
        else:
            mask = torch.ones_like(input_ids)
            pad = torch.zeros(
                [self.max_sent_len-len(input_ids)], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=0)
            mask = torch.cat([mask, pad], dim=0)

        results['refer'] = input_ids
        results['r_mask'] = mask
        return results


@PIPELINES.register_module()
class RandomParaPhrase:
    def __init__(self, phrase_cache, ratio=0.5) -> None:
        self.ratio = ratio
        with open(phrase_cache, 'r') as f:
            self.phrase_cache = json.load(f)

    def __call__(self, results):
        if np.random.random() >= self.ratio:
            name = results['img_info']['file_name']
            cache = self.phrase_cache[name]
            phrase_num = len(cache)
            results['refer'] = cache[np.random.randint(
                0, phrase_num)].replace('?', '').lower()
        return results
