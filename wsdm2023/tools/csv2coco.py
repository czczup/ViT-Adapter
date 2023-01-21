import json
import pandas as pd
import datetime


def load_dataset(name):
    csv_path = f'data/wsdm2023/annotations/{name}.csv'
    dataset = pd.read_csv(csv_path)

    # img_path = f'/home/data2/gaoshengyi/datasets/wsdm2023/{name}'
    # if not os.path.exists(img_path):
    #     os.mkdir(img_path)
    # for img_url in tqdm(dataset.image):
    #     img_name = img_url.split('/')[-1]
    #     if not os.path.exists(f'{img_path}/{img_name}'):
    #         wget.download(img_url, out=img_path)

    anno_path = f'data/wsdm2023/annotations/{name}.json'
    info = {
        "description": "WSDMCUP2023 dataset",
        "url": "https://toloka.ai/challenges/wsdm2023/",
        "version": "1.0",
        "year": 2022,
        "contributor": "toloka",
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    licenses = []  # no license
    images = []
    for idx, data in dataset.iterrows():
        url = data['image']
        img_info = {
            "coco_url": url,
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": url.split('/')[-1],
            "flickr_url": "",
            "id": idx,
            "license": 0,
            "width": data['width'],
            "height": data['height']
        }
        images.append(img_info)
    annotations = []
    for idx, data in dataset.iterrows():
        x, y, w, h = data['left'], data['top'], data['right'] - \
            data['left'], data['bottom']-data['top']
        bbox_info = {
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "segmentation": [[x, y, x+w, y, x+w, y+h, x, y+h]],
            "area": w * h,
            "bbox": [] if name == 'test_public' else [x, y, w, h],
            "iscrowd": 0
        }
        annotations.append(bbox_info)
    categories = [{
        "id": 1,
        "name": "object",
        "supercategory": "object",
    }]
    # add question to annotation
    questions = []
    for idx, data in dataset.iterrows():
        question_info = {
            "id": idx,
            "image_id": idx,
            "question": data['question']
        }
        questions.append(question_info)
    anno = {
        "info": info,
        "lisences": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "type": 'instances',
        "questions": questions
    }
    anno = json.dumps(anno)
    with open(anno_path, 'w') as f:
        f.write(anno)


def main():
    load_dataset('train_sample')
    load_dataset('train')
    load_dataset('test_public')
    load_dataset('val')


if __name__ == '__main__':
    main()
