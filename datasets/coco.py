"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as TT
import os
import random
from PIL import Image
import numpy as np
from PIL import ImageEnhance


class CocoDetection(TvCocoDetection):
    def __init__(self,
                 img_folder,
                 ann_file,
                 transforms,
                 return_masks,
                 cache_mode=False,
                 local_rank=0,
                 local_size=1,
                 support_frms=True,
                 num_global_frames=3,
                 shuffled_aug=None):
        super(CocoDetection, self).__init__(img_folder,
                                            ann_file,
                                            cache_mode=cache_mode,
                                            local_rank=local_rank,
                                            local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.support_frms = support_frms
        self.num_global_frames = num_global_frames
        self.shuffled_aug = shuffled_aug

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        if self.support_frms:
            path = self.coco.loadImgs(image_id)[0]['file_name']
            path_name_id = int(path[-10:-4])
            path_filetyp = path[-4:]
            path_dirname = os.path.dirname(path)
            max_frm_id = len(os.listdir(os.path.join(self.root,
                                                     path_dirname))) - 1
            supp_frms_id = [path_name_id - 1,
                            path_name_id + 1]  # pre and after
            supp_frms_id = [min(max(i, 0), max_frm_id) for i in supp_frms_id]
            supp_frms = [
                self.get_image(f'{path_dirname}/{i:06d}{path_filetyp}')
                for i in supp_frms_id
            ]

            # get the global frames
            image_glbs = []
            if self.num_global_frames > 0:

                select_range = [0, max_frm_id]
                select_candidate = list(range(*select_range))
                if path_name_id in select_candidate:
                    select_candidate.remove(path_name_id)
                # while True:
                glb_file_id = random.sample(select_candidate,
                                            self.num_global_frames)
                global_frms = [
                    self.get_image(f'{path_dirname}/{i:06d}{path_filetyp}')
                    for i in glb_file_id
                ]

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # to tensor
        img = [img] + supp_frms if self.support_frms else []

        # suffled video augmentation
        if self.shuffled_aug is not None:
            global_frms, target_ = shuffled_aug(global_frms, target,
                                                self.shuffled_aug)
            global_frms, target_ = T.resize(global_frms, target, img[0].size,
                                            1333)

        img = img + global_frms

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def shuffled_aug(global_frms, target, type='centerCrop'):
    # print('applying {} on shuffled video...'.format(type))
    if type == 'centerCrop':
        return CenterCrop(global_frms, target)
    elif type == 'hflip':
        return T.hflip(global_frms, target)
    elif type == 'RandomSizeCrop':
        return RandomSizeCrop(global_frms, target)
    elif type == 'randomPeper':
        return [randomPeper(img) for img in global_frms], target
    elif type == 'vflip':
        return [F.vflip(img) for img in global_frms], target
    elif type == 'randomRotation':
        return [randomRotation(img) for img in global_frms], target
    elif type == 'colorEnhance':
        return [colorEnhance(img) for img in global_frms], target
    else:
        return global_frms, target


def randomRotation(image):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        # label = label.rotate(random_angle, mode)
    return image


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    # print(img.shape)
    width, height, c = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height, c])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


def CenterCrop(img, target):

    if isinstance(img, list):
        image_width, image_height = img[0].size
        crop_height, crop_width = 400, 400
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))

        return T.crop(img, target,
                      (crop_top, crop_left, crop_height, crop_width))


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return T.crop(img, target, region)


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


def RandomSizeCrop(img, target):
    if isinstance(img, list):
        w = random.randint(400, min(img[0].width, 1333))
        h = random.randint(400, min(img[0].height, 1333))
        region = TT.RandomCrop.get_params(img[0], [h, w])
    else:
        w = random.randint(400, min(img.width, 1333))
        h = random.randint(400, min(img.height, 1333))
        region = TT.RandomCrop.get_params(img, [h, w])
    return T.crop(img, target, region)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [
            obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0
        ]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576]
    strong_aug = False
    if image_set == 'train':
        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500]),
                        T.RandomSizeCrop(384, 512),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                SLT.RandomSelectMulti([
                    # SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),              
                normalize,
            ])

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([640]),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([640], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_coco_lite_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    data_mode = args.data_mode
    PATHS = {
        "train":
        (root / "rawframes", root / f'imagenet_vid_train_{data_mode}.json'),
        "val": (root / "rawframes", root / f'imagenet_vid_val.json')
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder,
                            ann_file,
                            transforms=make_coco_transforms(image_set),
                            return_masks=args.masks,
                            cache_mode=args.cache_mode,
                            local_rank=get_local_rank(),
                            local_size=get_local_size())
    return dataset
