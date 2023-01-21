import pandas as pd
from mmdet.apis import init_detector
import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.ops import RoIPool
import argparse
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403


def multimodel_inference(model, imgs, questions):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        questions=[questions]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img,question in zip(imgs,questions):
        # add information into dict
        data = dict(img_info=dict(filename=img,question=question), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


def main(dataset, config_file, checkpoint_file,device='cuda:0'):
    model=init_detector(config=config_file,
                        checkpoint=checkpoint_file,
                        device=device)
    ann=pd.read_csv(f'data/wsdm2023/annotations/{dataset}.csv')
    data_root=(f'data/wsdm2023/{dataset}/')
    
    for idx,data in ann.iterrows():
        img_url=data['image']
        img_name=img_url.split('/')[-1]
        res=multimodel_inference(model,data_root+img_name,data['question'])
        print(res)
        
        
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument('config',type=str,default='')
    parser.add_argument('checkpoint',type=str,default='')
    parser.add_argument('dataset',type=str,default='')
    parser.add_argument('--device',type=str,default='cuda:0')
    
    args=parser.parse_args()
    main(args.dataset,args.config,args.checkpoint,args.device)