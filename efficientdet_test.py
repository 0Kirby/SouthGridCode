import os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import argparse

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import json

from glob import glob
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch Test')
    parser.add_argument('--img_path', type=str, default='/home/kesci/input/test3132/test/image/',  help='test image path')
    parser.add_argument('--weights', type=str, default='weights/efficientdet-d2.pth', help='load weights path')
    parser.add_argument('--img_id', type=int, default=40, help='test image id')
    args = parser.parse_args()
    return args


def test(opt):
    compound_coef = 2
    force_input_size = None  # set None to use default size
    img_id = opt.img_id
    img_path = opt.img_path
    img_path = img_path + str(img_id) + '.jpg'

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.2
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ['02010001', '02010002']

    color_list = standard_to_bgr(STANDARD_COLORS)
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(opt.weights, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    def display(preds, imgs, imshow=True, imwrite=False, img_id=1):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            imgs[i] = imgs[i].copy()
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)

            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                             color=color_list[get_index_label(obj, obj_list)])

            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if imwrite:
    
                str1 = 'test/' + str(img_id) + '.jpg'
                cv2.imwrite(str1, imgs[i])

    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=True , img_id=img_id)

    print('running speed test...')
    with torch.no_grad():
        print('test1: model inferring and postprocessing')
        print('inferring image for 10 times...')
        t1 = time.time()
        for _ in range(10):
            _, regression, classification, anchors = model(x)
            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            out = invert_affine(framed_metas, out)
        tempList = []
        for j in range(len(out[0]['class_ids'])):
            tempout = {}
            tempout['image_id'] = img_id
            if out[0]['class_ids'][j] == 1:
                tempout['category_id'] = 2
            else:
                tempout['category_id'] = 1
            tempout['score'] = out[0]['scores'][j].astype(np.float64)
            tempout['bbox'] = [(out[0]['rois'][j][0]).astype(np.float64), (out[0]['rois'][j][1]).astype(np.float64),
                               (out[0]['rois'][j][2]).astype(np.float64) - (out[0]['rois'][j][0]).astype(np.float64),
                               (out[0]['rois'][j][3]).astype(np.float64) - (out[0]['rois'][j][1]).astype(np.float64),
                               ]
            tempList.append(tempout)
        t2 = time.time()
        tact_time = (t2 - t1) / 10
        print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')
        with open("test/" + str(img_id) + ".json", "w") as f:
            json.dump(tempList, f)
        print("生成标注后的图片("+str(img_id)+".jpg)和json("+str(img_id)+".json)到test文件夹中...")


if __name__ == '__main__':
    opt = get_args()
    test(opt)

