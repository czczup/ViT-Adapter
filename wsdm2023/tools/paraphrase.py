import torch
from parrot import Parrot
import json
import pandas
import argparse
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='csv file path')
    parser.add_argument('out', type=str, help='output json file path')
    parser.add_argument('--topn', type=int, default=3,
                        help='use top n paraphrase for augment')

    return parser.parse_args()


def main(args):
    parrot = Parrot(
        model_tag="prithivida/parrot_paraphraser_on_T5")
    parrot.model = parrot.model.to('cuda:0')
    print('Successfully load model.')
    res = dict()

    df = pandas.read_csv(args.csv)
    total = len(df)
    for idx, data in df.iterrows():
        name = data['image'].split('/')[-1]
        phrase = data['question'].replace(
            '\"', '').replace('?', '').strip(' ').lower()
        paras = parrot.augment(input_phrase=phrase, use_gpu=True)
        print('-'*100)
        print(phrase)
        print('-'*100)
        print(paras)
        if paras is None:
            res[name] = [phrase]
        else:
            selected = []
            for i, p in enumerate(paras):
                selected.append(p[0])
                if i >= args.topn:
                    break
            res[name] = selected

        print(f'Finished [{idx+1}/{total}]\n')

    with open(args.out, 'w') as f:
        res = json.dumps(res)
        f.write(res)


if __name__ == '__main__':
    main(parse_args())
