import glob, os
import argparse
import yaml

import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

def display(preds, imgs, outdir, files, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

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
            out_name = outdir + files[i]
            cv2.imwrite(out_name, imgs[i])

def savepreds(preds, savedir, files):
    kitti_string = '%s %.2f %.0f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'

    for i in range(len(files)):
        if len(preds[i]['rois']) == 0:
            continue

        name = files[i]
        label_file = open(savedir + name[0: -3] + 'txt', 'a')

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            label_string = kitti_string % (obj, score, 0, 0, x1, y1, x2, y2, 0, 0, 0, 0, 0, 0, 0)
            label_file.write(label_string)

        label_file.close()

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

parser = argparse.ArgumentParser('Neurala test EfficientDet Pytorchparser')
parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
parser.add_argument('-b', '--batch_size', type=int, default=10, help='inference batch size')
parser.add_argument('-s', '--save_preds', type=bool, default=False, help='Saves detections to text file')
parser.add_argument('--show_images', type=bool, default=False, help='Shows images as they are processed')
parser.add_argument('--save_images', type=bool, default=False, help='Saves images to output directory')
args = parser.parse_args()

params = Params(f'projects/{args.project}.yml')
compound_coef = args.compound_coef

force_input_size = None  # set None to use default size

img_path = params.val_data_path + 'images/'         #'test/img.png'
anchor_ratios = params.anchors_ratios        #[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = params.anchors_scales        #[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = params.threshold
iou_threshold = params.iou_threshold
obj_list = params.obj_list
out_dir = params.outpud_dir

cudnn.benchmark = True
cudnn.fastest = True
use_cuda = True
use_float16 = False

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
    ratios=eval(anchor_ratios), scales=eval(anchor_scales))
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}-{args.project}.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

images = []
names = []
for ext in ('*.gif', '*.png', '*.jpg', '*.PNG', '*.JPG'):
    image_list = glob.glob(img_path + ext)
    
    if(len(image_list) == 0):
        continue
    
    last_file = image_list[-1]

    for image in image_list:
        images.append(image);
        names.append(os.path.basename(image))

        if len(images) >= args.batch_size or image == last_file:

            ori_imgs, framed_imgs, framed_metas = preprocess(images, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)

                out = invert_affine(framed_metas, out)
                display(out, ori_imgs, out_dir, names, imshow=args.show_images, imwrite=args.save_images)

                if args.save_preds:
                    if isinstance(params.preds_path, type(None)):
                        save_preds_path = params.val_data_path + 'neupreds/'
                    else:
                        save_preds_path = params.preds_path

                    os.makedirs(save_preds_path, exist_ok=True)
                    savepreds(out, save_preds_path, names)

            images = []
            names = []
