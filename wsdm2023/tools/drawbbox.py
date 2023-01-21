import cv2
import argparse
import torch
import json
from torchvision.utils import draw_bounding_boxes
from torch.utils.tensorboard import SummaryWriter


def xywh2xyxy(bbox):
    x, y, w, h = [int(val) for val in bbox]
    return [x, y, x+w, y+h]


def draw_bboxes(img_url, pred, gt, args):
    img = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1)

    if len(gt) == 0:
        pred_bb = torch.tensor(xywh2xyxy(pred)).reshape(1, -1)
        img = draw_bounding_boxes(img, pred_bb, labels=['pred'], colors=[
            (255, 0, 0)], width=5, font_size=10)
    else:
        pred_bb = torch.tensor(xywh2xyxy(pred)).reshape(1, -1)
        gt_bb = torch.tensor(xywh2xyxy(gt)).reshape(1, -1)
        img = draw_bounding_boxes(img, torch.cat([pred_bb, gt_bb], dim=0), labels=[
            'pred', 'gt'], colors=[(255, 0, 0), (0, 255, 0)], width=5, font_size=10)

    return img


def main(args):
    with open(f'data/wsdm2023/annotations/{args.data}.json', 'r') as load_f:
        load_dict = json.load(load_f)
        images = load_dict['images']
        annotations = load_dict['annotations']

    with open(args.result, 'r') as load_f:
        load_dict = json.load(load_f)

    img_index = dict()
    for img in images:
        img_index[img['id']] = img['coco_url'].split('/')[-1]

    ann_index = dict()
    for ann in annotations:
        ann_index[ann['image_id']] = ann['bbox']

    with SummaryWriter(args.output) as writer:
        cnt = 0
        for res in load_dict:
            id = res['image_id']
            bbox = res['bbox']
            img_url = f'data/wsdm2023/{args.data}/{img_index[id]}'
            gt = ann_index[id]

            img = draw_bboxes(img_url, bbox, gt, args)
            writer.add_image(img_index[id], img)
            cnt += 1
            print(
                f'Add {img_url} to tensorboard {args.output}, finished [{cnt}/{len(load_dict)}].')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('result', type=str)
    parser.add_argument('--output', type=str,
                        default='./runs/test_public')

    args = parser.parse_args()
    main(args)
