from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset
from math import sqrt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import deque
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import load_model

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import matplotlib

# Use the 'Agg' backend for matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio

glbl_count = 1

activity_data = []

sequence_buffer = []


cache_dir = os.getenv('CACHE_DIR', '/mnt/cache-dir')


model_path = os.getenv('LSTM_MODEL_PATH', '/mnt/lstm-model-directory')


base_directory = os.getenv('SAVE_BASE_DIRECTORY', '/mnt/ave-directory')

# Construct the full path
save_directory = os.path.join(base_directory, file_name)


sequence_length = 5  # Define the number of timesteps to consider for each prediction
num_features_per_keypoint = 3  # Assuming each frame's features consist of 'relative_x', 'relative_y', 'score'

# Initialize a deque to hold the most recent sequence_length frames' features
features_buffer = deque(maxlen=sequence_length)


def extract_features(frame):
    # Define a list of expected keypoint IDs as strings
    expected_ids = [str(i) for i in range(1, 18)]  # Adjust the range as per your keypoints IDs
    
    # Initialize an empty list to hold features
    keypoints_features = []
    
    # Iterate through the expected keypoint IDs
    for expected_id in expected_ids:
        # Check if the current keypoint ID is in the frame's keypoints
        if expected_id in frame:
            # Key point is present, extract features
            keypoints_features.extend([frame[expected_id]['relative_x'], frame[expected_id]['relative_y'], frame[expected_id]['score']])
        else:
            # Key point is missing, add default values
            # It looks like there might have been a typo in your original snippet (-1 repeated five times instead of three).
            # Adjust the number of -1s based on the actual number of features you have per keypoint.
            keypoints_features.extend([-1, -1, -1])  # Use three -1 values assuming each keypoint should have three features
    
    return keypoints_features






def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})




os.environ['TFHUB_CACHE_DIR'] = cache_dir

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
MIN_CROP_KEYPOINT_SCORE = 0.2

def loadModel():
    model_name = "movenet_thunder"  # Replace this with your desired model name
    
    if "tflite" in model_name:
        # Code for handling .tflite models
        if "movenet_lightning_f16" in model_name:
            # Your code to download the movenet_lightning_f16 model
            input_size = 192
        elif "movenet_thunder_f16" in model_name:
            # Your code to download the movenet_thunder_f16 model
            input_size = 256
        elif "movenet_lightning_int8" in model_name:
            # Your code to download the movenet_lightning_int8 model
            input_size = 192
        elif "movenet_thunder_int8" in model_name:
            # Your code to download the movenet_thunder_int8 model
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        # Initialize the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter, input_size
    else:
        # Code for handling models from TensorFlow Hub
        if "movenet_lightning" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            input_size = 192
        elif "movenet_thunder" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)
        
        return module, input_size


module, input_size = loadModel()


LSTM_LOADED_MODEL = load_model(model_path)

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    global glbl_count

    print("GLOBAL COUNT!!" , glbl_count)
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    
    if args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    if glbl_count is not None:
        # Define font, size, and color for the frame number text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 2
        cv2.putText(img_numpy, f'Frame: {glbl_count}', (10, 20), font, font_scale, color, thickness)

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            

    return img_numpy

def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()
    
    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)
        

        

def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h*w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes  , gt_boxes   = split(gt_boxes)
                crowd_masks  , gt_masks   = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h*w).cuda()
        boxes = boxes.cuda()


    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i,:],   box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], mask_scores[i])
            return
    
    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt   = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item(),
                     lambda i,j: crowd_bbox_iou_cache[i,j].item(),
                     lambda i: box_scores[i], box_indices),
            ('mask', lambda i,j: mask_iou_cache[i, j].item(),
                     lambda i,j: crowd_mask_iou_cache[i,j].item(),
                     lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)
                
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                
                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

def evalimage(net:Yolact, path:str, save_path:str=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
    
    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)

def evalimages(net:Yolact, input_folder:str, output_folder:str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'): 
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')

from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def evalvideo(net:Yolact, path:str, out_path:str=None):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()
    
    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True
    
    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()



    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames


    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            print("FRAMES!!",frames)

            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp

            print("FRAMES!! ", frames)

            print("imgs!!", imgs)

            print("imgs type!!", type(imgs))
            num_extra = 0
            while imgs.size(0) < args.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out
    def normalize_keypoints_relative_to_hip(keypoints_with_scores):
        # Assuming the hip keypoint indexes are known, e.g., right_hip_index and left_hip_index
        right_hip_index = 12  # Example index, adjust based on your model's keypoint definition
        left_hip_index = 11  # Example index
        
        # Extract hip coordinates
        right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
        left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]
        hip_center = (right_hip + left_hip) / 2

        # Initialize dictionary to store normalized keypoints
        normalized_keypoints = {}

        # Iterate through all keypoints to calculate relative positions
        for i, keypoint in enumerate(keypoints_with_scores[0, 0, :, :2]):
            relative_position = keypoint - hip_center
            normalized_keypoints[i] = {
                'relative_x': relative_position[1],  # x-coordinate
                'relative_y': relative_position[0],  # y-coordinate
                'absolute_x': keypoint[1],  # Original x-coordinate
                'absolute_y': keypoint[0],  # Original y-coordinate
                'score': keypoints_with_scores[0, 0, i, 2]  # Confidence score
            }

        return normalized_keypoints

    def init_crop_region(image_height, image_width):
      """Defines the default crop region.

      The function provides the initial crop region (pads the full image from both
      sides to make it a square image) when the algorithm cannot reliably determine
      the crop region from the previous frame.
      """
      if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
      else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

      return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
      }
    def crop_and_resize(image, crop_region, crop_size):
      """Crops and resize the image to prepare for the model input."""
      boxes=[[crop_region['y_min'], crop_region['x_min'],
              crop_region['y_max'], crop_region['x_max']]]
      output_image = tf.image.crop_and_resize(
          image, box_indices=[0], boxes=boxes, crop_size=crop_size)
      return output_image
    def run_inference(movenet, image, crop_region, crop_size):
      """Runs model inference on the cropped region.

      The function runs the model inference on the cropped region and updates the
      model output to the original image coordinate system.
      """
      image_height, image_width, _ = image.shape
      input_image = crop_and_resize(
        tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
      # Run model inference.
      keypoints_with_scores = movenet(input_image)
      # Update the coordinates.
      for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0, 0, idx, 1]) / image_width
      return keypoints_with_scores
    def movenet(input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        model = module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores
    def normalize_keypoints_relative_to_hip(keypoints_with_scores, score_threshold=0.5):
        # Assuming the hip keypoint indexes are known, e.g., right_hip_index and left_hip_index
        right_hip_index = 12  # Example index, adjust based on your model's keypoint definition
        left_hip_index = 11  # Example index
        
        # Extract hip coordinates
        right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
        left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]
        hip_center = (right_hip + left_hip) / 2

        # Initialize dictionary to store normalized keypoints
        normalized_keypoints = {}

        # Iterate through all keypoints to calculate relative positions
        for i, keypoint in enumerate(keypoints_with_scores[0, 0, :, :2]):
            score = keypoints_with_scores[0, 0, i, 2]  # Confidence score
            
            # Check if the keypoint's confidence score exceeds the threshold
            if score > score_threshold:
                relative_position = keypoint - hip_center
                normalized_keypoints[i] = {
                    'relative_x': relative_position[1],  # x-coordinate
                    'relative_y': relative_position[0],  # y-coordinate
                    'absolute_x': keypoint[1],  # Original x-coordinate
                    'absolute_y': keypoint[0],  # Original y-coordinate
                    'score': score  # Confidence score
                }

        return normalized_keypoints


    def draw_prediction_on_image(
        image, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
      """Draws the keypoint predictions on image.

      Args:
        image: A numpy array with shape [height, width, channel] representing the
          pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
          the keypoint coordinates and scores returned from the MoveNet model.
        crop_region: A dictionary that defines the coordinates of the bounding box
          of the crop region in normalized coordinates (see the init_crop_region
          function below for more detail). If provided, this function will also
          draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
          Note that the image aspect ratio will be the same as the input image.

      Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions.
      """
      height, width, channel = image.shape
      aspect_ratio = float(width) / height
      fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
      # To remove the huge white borders
      fig.tight_layout(pad=0)
      ax.margins(0)
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      plt.axis('off')

      im = ax.imshow(image)
      line_segments = LineCollection([], linewidths=(4), linestyle='solid')
      ax.add_collection(line_segments)
      # Turn off tick labels
      scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

      (keypoint_locs, keypoint_edges,
       edge_colors) = _keypoints_and_edges_for_display(
           keypoints_with_scores, height, width)

      line_segments.set_segments(keypoint_edges)
      line_segments.set_color(edge_colors)
      if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
      if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

      if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(
          fig.canvas.get_width_height()[::-1] + (3,))
      plt.close(fig)
      if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
             interpolation=cv2.INTER_CUBIC)
      return image_from_plot
    def _keypoints_and_edges_for_display(keypoints_with_scores,
                                         height,
                                         width,
                                         keypoint_threshold=0.11):
      """Returns high confidence keypoints and edges for visualization.

      Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
          the keypoint coordinates and scores returned from the MoveNet model.
        height: height of the image in pixels.
        width: width of the image in pixels.
        keypoint_threshold: minimum confidence score for a keypoint to be
          visualized.

      Returns:
        A (keypoints_xy, edges_xy, edge_colors) containing:
          * the coordinates of all keypoints of all detected entities;
          * the coordinates of all skeleton edges of all detected entities;
          * the colors in which the edges should be plotted.
      """
      keypoints_all = []
      keypoint_edges_all = []
      edge_colors = []
      num_instances, _, _, _ = keypoints_with_scores.shape
      for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
          if (kpts_scores[edge_pair[0]] > keypoint_threshold and
              kpts_scores[edge_pair[1]] > keypoint_threshold):
            x_start = kpts_absolute_xy[edge_pair[0], 0]
            y_start = kpts_absolute_xy[edge_pair[0], 1]
            x_end = kpts_absolute_xy[edge_pair[1], 0]
            y_end = kpts_absolute_xy[edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)
      if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
      else:
        keypoints_xy = np.zeros((0, 17, 2))

      if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
      else:
        edges_xy = np.zeros((0, 2, 2))
      return keypoints_xy, edges_xy, edge_colors
    def find_closest_keypoint(keypoints_with_scores, score_threshold):
        # Initialize maximum score and reference keypoint
        max_score = 0
        keypoint_reference = None
        
        # Iterate over all keypoints to find the one with the highest score above the threshold
        for i, keypoint_data in enumerate(keypoints_with_scores[0, 0]):
            score = keypoint_data[2]
            if score > max_score and score > score_threshold:
                max_score = score
                keypoint_reference = keypoint_data[:2]  # x, y coordinates of the keypoint
        
        return keypoint_reference

    def calculate_object_distances_from_keypoint(best_detections_by_class, keypoints_with_scores, score_threshold=0.5):
        # Assuming hip indexes as before
        right_hip_index = 12
        left_hip_index = 11
        right_hip_score = keypoints_with_scores[0, 0, right_hip_index, 2]
        left_hip_score = keypoints_with_scores[0, 0, left_hip_index, 2]
        
        # Check if the scores for the hip keypoints are above the threshold
        if right_hip_score > score_threshold and left_hip_score > score_threshold:
            # If both hips are detected, calculate the center
            right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
            left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]
            keypoint_reference = (right_hip + left_hip) / 2
        else:
            # If hips are not detected, find the closest keypoint above the score threshold
            keypoint_reference = find_closest_keypoint(keypoints_with_scores, score_threshold)
            if keypoint_reference is None:
                # If no keypoints are detected above the threshold, return None
                return None

        # Create a dictionary to hold class names and distances from the keypoint
        class_distances = {}

        # Calculate the distance of each object's centroid from the keypoint_reference
        for class_name, data in best_detections_by_class.items():
            centroid = data['centroid']
            distance = sqrt((centroid[0] - keypoint_reference[0]) ** 2 + (centroid[1] - keypoint_reference[1]) ** 2)
            # Store only the class name and the distance in the new dictionary
            class_distances[class_name] = distance

        return class_distances

# Usage:
# keypoints_with_scores should be the output from a pose estimation model
# best_detections_by_class should be the output from the object detection model
#distances_from_keypoint = calculate_object_distances_from_keypoint(best_detections_by_class, keypoints_with_scores, score_threshold=0.5)
    def extract_relevant_data3(preds, score_threshold=0.5):
        class_names = cfg.dataset.class_names
        best_detections_by_class = {}

        detections = preds[0]['detection']
        boxes = detections['box']
        masks = detections['mask']
        classes = detections['class']
        scores = detections['score']

        for box, mask, cls, score in zip(boxes, masks, classes, scores):
            if score > score_threshold:
                class_name = class_names[cls.item()]

                # Convert box and mask to list for JSON serialization
                box_list = box.cpu().numpy().tolist()
                mask_list = mask.cpu().numpy().tolist()

                centroid = [(box_list[0] + box_list[2]) / 2, (box_list[1] + box_list[3]) / 2]

                if class_name not in best_detections_by_class or score.item() > best_detections_by_class[class_name]['score']:
                    # Update the best (highest scoring) detection for the class
                    best_detections_by_class[class_name] = {
                        'box': box_list,
                        'mask': mask_list,
                        'score': score.item(),
                        'centroid': centroid
                    }

        # Calculate distances between the best objects of different classes
        class_combinations_checked = set()
        for class_1, data_1 in best_detections_by_class.items():
            for class_2, data_2 in best_detections_by_class.items():
                if class_1 != class_2 and (class_2, class_1) not in class_combinations_checked:
                    distance = sqrt((data_2['centroid'][0] - data_1['centroid'][0]) ** 2 + (data_2['centroid'][1] - data_1['centroid'][1]) ** 2)
                    if 'distances' not in best_detections_by_class[class_1]:
                        best_detections_by_class[class_1]['distances'] = {}
                    best_detections_by_class[class_1]['distances'][class_2] = distance

                    if 'distances' not in best_detections_by_class[class_2]:
                        best_detections_by_class[class_2]['distances'] = {}
                    best_detections_by_class[class_2]['distances'][class_1] = distance

                    class_combinations_checked.add((class_1, class_2))

        return best_detections_by_class
    # def extract_relevant_data3(preds, score_threshold=0.5):
    #     class_names = cfg.dataset.class_names
    #     detections_by_class = {}

    #     detections = preds[0]['detection']
    #     boxes = detections['box']
    #     masks = detections['mask']
    #     classes = detections['class']
    #     scores = detections['score']

    #     for box, mask, cls, score in zip(boxes, masks, classes, scores):
    #         if score > score_threshold:
    #             class_name = class_names[cls.item()]

    #             # Convert box and mask to list for JSON serialization
    #             box_list = box.cpu().numpy().tolist()
    #             mask_list = mask.cpu().numpy().tolist()

    #             centroid = [(box_list[0] + box_list[2]) / 2, (box_list[1] + box_list[3]) / 2]

    #             if class_name not in detections_by_class:
    #                 detections_by_class[class_name] = {
    #                     'boxes': [],
    #                     'masks': [],
    #                     'scores': [],
    #                     'centroids': [],
    #                     'distances': None  # Initialize with None
    #                 }

    #             detections_by_class[class_name]['boxes'].append(box_list)
    #             detections_by_class[class_name]['masks'].append(mask_list)
    #             detections_by_class[class_name]['scores'].append(score.item())
    #             detections_by_class[class_name]['centroids'].append(centroid)

    #     # Calculate distances between objects of the same class
    #     for class_name, data in detections_by_class.items():
    #         centroids = data['centroids']
    #         distances = []
    #         if len(detections_by_class.items()) > 1:
    #             print("FOUND MORE THAN 22!!")
    #             print(len(detections_by_class.items()))

    #         if len(centroids) > 1:
    #             for i in range(len(centroids) - 1):
    #                 for j in range(i + 1, len(centroids)):
    #                     distance = sqrt((centroids[j][0] - centroids[i][0]) ** 2 + (centroids[j][1] - centroids[i][1]) ** 2)
    #                     distances.append(distance)
    #             detections_by_class[class_name]['distances'] = distances
    #             print("detections_by_class ", detections_by_class[class_name]['distances'])
    #             print("detections_*******", detections_by_class[class_name])
    #         else:
    #             detections_by_class[class_name]['distances'] = None

    #     return detections_by_class

    # def extract_relevant_data3(preds, score_threshold=0.5):
    #     """
    #     Organize detections by class name into a nested dictionary structure.
        
    #     Args:
    #         preds: The predictions from the model.
    #         score_threshold (float): The threshold for filtering out low-confidence detections.
            
    #     Returns:
    #         A dictionary where each key is a class name and each value is another dictionary with keys 'boxes', 'masks', 'scores'.
    #     """
    #     # Assuming cfg is accessible and contains class names mapping, like cfg.dataset.class_names
    #     class_names = cfg.dataset.class_names  # This should be a list or dict mapping class indices to names

    #     detections_by_class = {}

    #     detections = preds[0]['detection']
    #     boxes = detections['box']
    #     masks = detections['mask']
    #     classes = detections['class']
    #     scores = detections['score']

    #     for box, mask, cls, score in zip(boxes, masks, classes, scores):
    #         if score > score_threshold:
    #             class_name = class_names[cls.item()]

    #             if class_name not in detections_by_class:
    #                 detections_by_class[class_name] = {'boxes': [], 'masks': [], 'scores': []}

    #             detections_by_class[class_name]['boxes'].append(box.cpu().numpy())
    #             detections_by_class[class_name]['masks'].append(mask.cpu().numpy())
    #             detections_by_class[class_name]['scores'].append(score.item())

    #     return detections_by_class

    def extract_relevant_data2(preds, score_threshold=0.5):
        # Assuming cfg is accessible and contains class names mapping, like cfg.dataset.class_names
        class_names = cfg.dataset.class_names  # This should be a list or dict mapping class indices to names

        filtered_boxes = []
        filtered_masks = []
        filtered_classes = []
        filtered_scores = []
        filtered_class_names = []  # For storing human-readable class names

        detections = preds[0]['detection']
        boxes = detections['box']
        masks = detections['mask']
        classes = detections['class']
        scores = detections['score']

        for box, mask, cls, score in zip(boxes, masks, classes, scores):
            if score > score_threshold:
                filtered_boxes.append(box.cpu().numpy())
                filtered_masks.append(mask.cpu().numpy())
                filtered_classes.append(cls.item())
                filtered_scores.append(score.item())
                filtered_class_names.append(class_names[cls.item()])  # Convert class index to human-readable name

        return filtered_boxes, filtered_masks, filtered_classes, filtered_scores, filtered_class_names


# Example usage
#score_threshold = 0.5
#filtered_boxes, filtered_masks, filtered_classes, filtered_scores = extract_relevant_data(preds, score_threshold=score_threshold)

    def convert_video_filename_to_filename(video_path):
        """
        Convert a video file path to a JSON filename by changing the extension to .json.
        The function returns only the JSON filename, without the full path.

        Args:
        video_path (str): The full path of the video file.

        Returns:
        str: The JSON filename corresponding to the video file.
        """
        import os

        # Extract the base name of the file (e.g., "IMG_1575.MOV")
        base_name = os.path.basename(video_path)
        # Split the base name into name and extension
        name, _ = os.path.splitext(base_name)
        # Append .json extension to the name
        json_filename = name
        
        return name


    def extract_relevant_data(preds, score_threshold=0.5):
        # Assume preds is a dictionary with keys 'detection', 'keypoints', etc.
        # Each key contains a list of detected items and their scores.
        detection = preds[0]

        print(type(detection))

        print(print(preds))
        print(type(preds))
        detections = preds[0]['detection']
        #keypoints = preds['keypoints']  # Assuming this structure exists


        print("DETECTION ", detections)


        for d in detections.keys():
            print("KEY!!")
            print(d)
        
        # Filter based on score threshold
        #filtered_boxes = [d['box'].cpu().numpy() for d in detections if d['score'] > score_threshold]

        print(filtered_boxes)
        #filtered_keypoints = [k for k, d in zip(keypoints, detections) if d['score'] > score_threshold]
        
        #return filtered_boxes, filtered_keypoints

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            global glbl_count
            global activity_data
            
            pbdframe = prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)


            score_threshold = 0.5
            
            #print(extract_relevant_data3(preds, score_threshold=score_threshold))

            #filtered_boxes, filtered_masks, filtered_classes, filtered_scores , filtered_class_names = extract_relevant_data2(preds, score_threshold=score_threshold)

            #print("FRAME NUMBER!! ", glbl_count)
            #print("FILTERED BOXES", filtered_boxes)

            #print("FILTERED_MASKS", filtered_masks)

            #print("FILTERED CLASES!!", filtered_classes)

            #print("FILTERED SCORES!!", filtered_scores)

            #print("FILTERED CLASS NAMES!!", filtered_class_names)

            #print(extract_relevant_data(preds))

            
            #image_rgb = Image.fromarray(pbdframe.astype('uint8','RGB'))
            frame_rgb = cv2.cvtColor(pbdframe, cv2.COLOR_BGR2RGB)


            print()

            image_height, image_width, _ = frame_rgb.shape


            crop_region = init_crop_region(image_height, image_width)



            keypoints_with_scores = run_inference(movenet, frame_rgb, crop_region, [input_size, input_size])



            print("keypoints with score ", keypoints_with_scores)
           # print("PREDS!! ", preds['detection'])
            print(type(preds))


            #for preds in preds_list:  # Iterate over each prediction in the list



#score_threshold = 0.5  # Example threshold

#filtered_boxes = [box for box, score in zip(boxes, scores) if score > score_threshold]
#filtered_masks = [mask for mask, score in zip(masks, scores) if score > score_threshold]


            #print(preds)

            #boxes = preds['loc'].cpu().numpy()
            print("BOXES!!")

            #print(boxes)


            print("KEYPOINTS WITH SCORES!!")

            print(keypoints_with_scores)


            print("keypoints with score ", keypoints_with_scores)


            print("NORMALIZED ",)



            annotated_image = draw_prediction_on_image(frame_rgb, keypoints_with_scores, crop_region=None, close_figure=True,output_image_height=image_height)
            file_name = convert_video_filename_to_filename(args.video)

            print("FILENAME!!", convert_video_filename_to_filename(args.video))
            # Define the directory path and filename
            filename = 'annotated_image'+str(glbl_count)+'.jpg'
            full_path = os.path.join(save_directory, filename)

            # Ensure the directory exists
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Convert the image from RGB to BGR format
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # Save the image
            cv2.imwrite(full_path, annotated_image_bgr)

            #print("EXTRACT DATA 3!#@#@ ",extract_relevant_data3(preds, score_threshold=score_threshold))
            #activity_data.append({
            #    "activity": "",  # or "swinging", etc.
            #    "frame_number": glbl_count,
            #    "normalized_keypoints":normalize_keypoints_relative_to_hip(keypoints_with_scores),
            #    "filtered_boxes":filtered_boxes,
            #    "filtered_class_names":filtered_class_names,
            #    "filtered_scores":filtered_scores,
            #    "filtered_classes":filtered_classes,
            #    "filtered_masks":filtered_masks
            #})
            normalizedKeypoints = normalize_keypoints_relative_to_hip(keypoints_with_scores)


            best_detections_by_class = extract_relevant_data3(preds, score_threshold=score_threshold)

            distances_from_keypoint = calculate_object_distances_from_keypoint(best_detections_by_class, keypoints_with_scores, score_threshold=0.5)
            print("DISTANCE FROM KEYPOINT!!", distances_from_keypoint)

            print("KEYPOINT TYPE!!!")

            print("keypoint type", type(normalizedKeypoints))

            print("Norm keypoints ", normalizedKeypoints)
            activity_data.append({
                "activity": "",
                "frame_number": glbl_count,
                "normalized_keypoints":normalizedKeypoints,
                "derection_objects":best_detections_by_class,
                "distance_from_keypoint":distances_from_keypoint

            })
            #print("activity DATA ", activity_data)
            print("image height ", image_height)
            print("image width ", image_width)
            glbl_count += 1
            return annotated_image

    frame_buffer = Queue()
    video_fps = 0
    def convert_video_filename_to_json_filename(video_path):
        """
        Convert a video file path to a JSON filename by changing the extension to .json.
        The function returns only the JSON filename, without the full path.

        Args:
        video_path (str): The full path of the video file.

        Returns:
        str: The JSON filename corresponding to the video file.
        """
        import os

        # Extract the base name of the file (e.g., "IMG_1575.MOV")
        base_name = os.path.basename(video_path)
        # Split the base name into name and extension
        name, _ = os.path.splitext(base_name)
        # Append .json extension to the name
        json_filename = name + ".json"
        
        return json_filename
    # def convert_numpy(obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()  # Convert ndarray to list
    #     elif isinstance(obj, np.float32):
    #         return float(obj)  # Convert float32 to Python native float
    #     elif isinstance(obj, dict):
    #         return {key: convert_numpy(value) for key, value in obj.items()}
    #     elif isinstance(obj, list):
    #         return [convert_numpy(item) for item in obj]
    #     else:
    #         return obj
    # def convert_numpy(obj):
    #     if isinstance(obj, np.generic):
    #         return np.asscalar(obj)  # Convert numpy scalar to Python native type
    #     elif isinstance(obj, np.ndarray):
    #         return obj.tolist()  # Convert ndarray to list
    #     elif isinstance(obj, dict):
    #         return {key: convert_numpy(value) for key, value in obj.items()}
    #     elif isinstance(obj, list):
    #         return [convert_numpy(item) for item in obj]
    #     else:
    #         return obj
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        elif isinstance(obj, np.generic):
            return obj.item()  # Convert numpy scalar to Python native type
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    # All this timing code to make sure that 
    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None:
                        print("CHECKING THE FRAME TYPE!! ", type(frame_buffer.get()))
                        cv2.imshow(path, frame_buffer.get())
                    else:
                        out.write(frame_buffer.get())
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                            % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                
                # This is split because you don't want savevideo to require cv2 display functionality (see #197)
                if out_path is None and cv2.waitKey(1) == 27:
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe
                
                if out_path is None or args.emulate_playback:
                    # This gives more accurate timing than if sleeping the whole amount at once
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    # Let's not starve the main thread, now
                    time.sleep(0.001)
        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()


    extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')






    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None: print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:
                frame_rate = vid.get(cv2.CAP_PROP_FPS)
                print("FRAME RATE!!! ", frame_rate)
                next_frames = pool.apply_async(get_next_frame, args=(vid,))

                print("NEXT FRAMES!!")
            else:
                next_frames = None
            
            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args =  [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
                
                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)
                
                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence)-1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0
            if fps == 0:
                print("VIDEO HAS FINISHED!!")
                print(args.video)
                output_file_path = 'activity_data.json'
                converted_activity_data = convert_numpy(activity_data)

                print("converted_activity_data",convert_video_filename_to_json_filename(args.video))
                file_name = convert_video_filename_to_json_filename(args.video)





                # Now you can safely serialize to JSON
                activity_data_json = json.dumps(converted_activity_data, indent=4)
                with open(file_name, 'w') as file:
                    file.write(activity_data_json)
                cleanup_and_exit()
            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (fps, video_fps, frame_buffer.qsize())
            if not args.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')
    
    cleanup_and_exit()

def evaluate(net:Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            evalvideo(net, inp, out)
        else:
            evalvideo(net, args.video)
        return

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    if not args.display and not args.benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))
    
    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

                # Test flag, do not upvote
                if cfg.mask_proto_debug:
                    with open('scripts/info.txt', 'w') as f:
                        f.write(str(dataset.ids[image_idx]))
                    np.save('scripts/gt.npy', gt_masks)

                batch = Variable(img.unsqueeze(0))
                if args.cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                preds = net(batch)
            # Perform the meat of the operation here depending on our mode.
            if args.display:
                img_numpy = prep_display(preds, img, h, w)
            elif args.benchmark:
                prep_benchmark(preds, h, w)
            else:
                prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)
            
            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())
            
            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy)
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
            elif not args.no_bar:
                if it > 1: fps = 1 / frame_times.get_avg()
                else: fps = 0
                progress = (it+1) / dataset_size * 100
                progress_bar.set_val(it+1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                    % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')



        if not args.display and not args.benchmark:
            print()
            if args.output_coco_json:
                print('Dumping detections...')
                if args.output_web_json:
                    detections.dump_web()
                else:
                    detections.dump()
            else:
                if not train_mode:
                    print('Saving data...')
                    with open(args.ap_data_file, 'wb') as f:
                        pickle.dump(ap_data, f)

                return calc_map(ap_data)
        elif args.benchmark:
            print()
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    print_maps(all_maps)
    
    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()



if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None        

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)

