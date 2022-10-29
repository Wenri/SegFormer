import glob
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from imageio.v2 import imsave

from mmseg.apis import init_segmentor
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmseg.core.evaluation import get_palette

from mmcv.parallel import collate, scatter


def inference_pipeline(cfg):
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    return test_pipeline


def segmentor(model, test_pipeline, **data):
    device = next(model.parameters()).device  # model device
    # prepare data
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    [img], [img_meta] = data['img'], data['img_metas']
    # forward the model
    with torch.no_grad():
        [seg_logit] = model.whole_inference(img, img_meta, rescale=True)
    return seg_logit.cpu().numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument('imgpath', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    pipeline = inference_pipeline(model.cfg)

    img_list = glob.glob(os.path.join(args.imgpath, '*.JPG'))
    palette = get_palette(args.palette)
    savepath = Path(args.imgpath).parent / 'seg'
    savepath.mkdir(exist_ok=True)

    for img in img_list:
        filename = Path(img)
        # test a single image
        seg_logit = segmentor(model, pipeline, img=img)
        print(filename.name, np.min(seg_logit), np.max(seg_logit))
        np.save(os.fspath(savepath / filename.with_suffix('.npy').name), seg_logit.astype(np.float16))

        seg_class = seg_logit.argmax(axis=0)
        seg_img = model.show_result(img, (seg_class,), palette=palette, show=False)
        # save the results
        imsave(savepath / filename.with_suffix('.png').name, seg_img)


if __name__ == '__main__':
    main()
