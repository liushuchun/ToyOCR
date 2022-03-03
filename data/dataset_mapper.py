import copy
from data.transforms.make_shrink_border import MakeShrinkMap
from data.transforms.make_border_map import MakeBorderMap
import logging

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from fvcore.common.file_io import PathManager
from PIL import Image
import cv2

from .transforms.arguement import arguementation

from . import transforms as T
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    logger = logging.getLogger("detectron2")

    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if max_size != 1:
        resize_type = cfg.INPUT.RESIZE_TYPE

        if resize_type == "ResizeShortestEdge":
            print(min_size, max_size)
            tfm_gens = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        elif resize_type == "Resize":
            try:
                min_size = min_size[0]
            except:
                min_size = int(min_size)
            tfm_gens = [T.Resize(shape=(min_size, max_size))]

    else:
        tfm_gens = []

    if cfg.MODEL.META_ARCHITECTURE == "CenterNet":
        if is_train:
            for (aug, args) in cfg.MODEL.DETNET.TRAIN_PIPELINES:
                tfm_gens.append(getattr(T, aug)(**args))
        else:
            for (aug, args) in cfg.MODEL.DETNET.TEST_PIPELINES:
                tfm_gens.append(getattr(T, aug)(**args))

    logger.info("TransformGens used: " + str(tfm_gens))

    return tfm_gens


def check_sample_valid(args):
    if args["sample_style"] == "range":
        assert (
            len(args["min_size"]) == 2
        ), f"more than 2 ({len(args['min_size'])}) min_size(s) are provided for ranges"


class DatasetMapper:
    """
    A callable which takes a dataset dict in textnet Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """
    def __init__(self, cfg, is_train=True):

        self.keep_size_and_crop = False
        self.db_keep_size_and_crop=False 
        self.text_det = False

        if cfg.INPUT.CROP.ENABLED and is_train and cfg.INPUT.CROP.TYPE != "crop_keep":
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE,
                                         cfg.INPUT.CROP.SIZE)
            logging.getLogger('detectron2').info("CropGen used in training: " +
                                                 str(self.crop_gen))
        elif cfg.INPUT.CROP.ENABLED and is_train and cfg.INPUT.CROP.TYPE == "crop_keep":
            self.keep_size_and_crop = True
            self.data_croper = T.RandomCropTransform()
            
        elif cfg.INPUT.CROP.ENABLED and is_train and cfg.INPUT.CROP.TYPE == "db_crop_keep":
            self.db_keep_size_and_crop = True 
            self.shrink_map_trans = MakeShrinkMap()
            self.border_map_trans=MakeBorderMap()
            
        else:
            self.crop_gen = None

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        self.tfm_gens = build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                                  if is_train else
                                  cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST)
        self.is_train = is_train
        self.BOX_MINSIZE = 1e-5
        self.imgaug_prob = 1.0

        if cfg.MODEL.META_ARCHITECTURE == "CenterNet":
            self.imgaug_prob = cfg.MODEL.DETNET.IMGAUG_PROB
            self.BOX_MINSIZE = cfg.MODEL.DETNET.BOX_MINSIZE

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in textnet Dataset format.

        Returns:
            dict: a format that builtin models in textnet accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"],
                                 format=self.img_format)
        # cv2.imshow("hello",image)
        # cv2.waitKey(2)
        utils.check_image_size(dataset_dict, image)

        origin_shape = image.shape[:2]
        
        
        if self.keep_size_and_crop:
            ignore_polys = [np.array(obj["poly"]).reshape(-1, 2) for obj in dataset_dict[
                "annotations"] if obj["ignore"] == 1]

            polys = [np.array(obj["poly"]).reshape(-1, 2) for obj in dataset_dict[
                "annotations"] if obj["ignore"] == 0]

            if ignore_polys:
                dataset_dict["ignore_polys"] = ignore_polys

            if "segm_file" in dataset_dict:
                with PathManager.open(dataset_dict.pop("segm_file"), "rb") as f:
                    segm_gt = Image.open(f)

                    segm_gt = np.asarray(segm_gt, dtype="uint8")


            mask = np.ones(origin_shape)

            if "ignore_polys" in dataset_dict:

                for poly in polys:
                    poly = np.array(poly, np.int32)
                    poly = poly.reshape(-1, 2)
                    cv2.fillPoly(mask, [poly], 0)

            image, segm_gt, mask = self.data_croper(
                image, polys, segm_gt, mask)

            dataset_dict["sem_seg"] = torch.as_tensor(
                segm_gt.astype("float32")/255.)

            dataset_dict["image"] = torch.as_tensor(
                image.transpose(2, 0, 1).astype("float32"))
            dataset_dict["mask"] = torch.as_tensor(mask)
            return dataset_dict


        if self.db_keep_size_and_crop:
            
            data_dict=self.data_croper(dataset_dict)

            polys = [
                np.array(obj["poly"]).reshape(-1, 2)
                for obj in dataset_dict["annotations"] if obj["ignore"] == 0
            ]

                
            text_polys= [
                np.array(obj["poly"]).reshape(-1, 2) for obj in dataset_dict["annotations"]]
            ignore_tags= [obj["ignore"] for obj in dataset_dict["annotations"]]

            data = dict(image=image, text_polys=text_polys, ignore_tags=ignore_tags)
            
            data = self.shrink_map_trans(data)
            data = self.border_map_trans(data)
            

            image, segm_gt, mask = self.data_croper(image, polys, data["gt"],
                                                    data["mask"])

            dataset_dict["sem_seg"] = torch.as_tensor(segm_gt)

            dataset_dict["image"] = torch.as_tensor(
                image.transpose(2, 0, 1).astype("float32"))
            dataset_dict["mask"] = torch.as_tensor(mask)
            return dataset_dict

        if "annotations" not in dataset_dict:

            print("with not crop")
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens,
                image)
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:

                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # apply imgaug
        if self.is_train and self.imgaug_prob < 1.0 and self.cfg.MODEL.META_ARCHITECTURE == "CenterNet":
            image = arguementation(image, self.imgaug_prob)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(dataset_dict, image_shape, transforms,
                                      self.min_box_side_len,
                                      self.proposal_topk)

        if not self.is_train and not self.eval_with_gt:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            ignore_polys = [
                obj["poly"] for obj in dataset_dict["annotations"]
                if obj["ignore"] == 1
            ]
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in dataset_dict.pop("annotations")
                if obj.get("ignore", 0) == 0
            ]

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format)
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(
                instances, box_threshold=self.BOX_MINSIZE)
            if ignore_polys:
                dataset_dict["ignore_polys"] = ignore_polys

        # USER: Remove if you don't do semantic  segmentation.

        if "segm_file" in dataset_dict:
            with PathManager.open(dataset_dict.pop("segm_file"), "rb") as f:
                segm_gt = Image.open(f)

                segm_gt = np.asarray(segm_gt, dtype="uint8")

                segm_shape = segm_gt.shape[0:2]

            segm_gt = transforms.apply_segmentation(segm_gt)

            segm_gt = torch.as_tensor(segm_gt.astype("float32") / 255.)

            dataset_dict["sem_seg"] = segm_gt

        mask = np.ones(origin_shape)

        if "ignore_polys" in dataset_dict:

            for poly in dataset_dict["ignore_polys"]:
                poly = np.array(poly, np.int32)
                poly = poly.reshape(-1, 2)
                cv2.fillPoly(mask, [poly], 0)

            # cv2.imshow("hi",np.array(mask,np.uint8)*255)
            # cv2.waitKey(0)
        mask = transforms.apply_segmentation(mask)
        mask = torch.as_tensor(mask)
        dataset_dict["mask"] = mask

        return dataset_dict
