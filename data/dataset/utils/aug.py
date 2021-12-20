
import math
import numbers
import random

import cv2
import numpy as np
from skimage.util import random_noise


class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """Random noise
        
        Args:
         data: {'image':,'text_polys':,'texts':,'ignore_tags':}
        return:
        """
        if random.random() > self.random_rate:
            return data
        data['image'] = (random_noise(data['image'], mode='gaussian', clip=True) * 255).astype(im.dtype)
        return data


class RandomScale:
    def __init__(self, scales, random_rate):
        """
        Args:
         scales: scale
         ramdon_rate: random coefficient
         
        """
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """random choose a scale
        
         data: {'image':,'text_polys':,'texts':,'ignore_tags':}
        Returns:
            dict:the processed data
        """
        if random.random() > self.random_rate:
            return data
        im = data['image']
        text_polys = data['text_polys']

        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale

        data['image'] = im
        data['text_polys'] = tmp_text_polys
        return data


class RandomRotateimageBox:
    def __init__(self, degrees, random_rate, same_size=False):
        """
         degrees:  the degree to rotate
         ramdon_rate: random rate
         same_size:  if keep the original size
        Returns:
            dict:the processed data
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """Random choose a scale
        
         data: {'image':,'text_polys':,'texts':,'ignore_tags':}
        Returns:
            dict:the processed data
        """
        if random.random() > self.random_rate:
            return data
        im = data['image']
        text_polys = data['text_polys']

        # ---------------------- 旋转图像 ----------------------
        w = im.shape[1]
        h = im.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            
            rangle = np.deg2rad(angle)
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_image = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        data['image'] = rot_image
        data['text_polys'] = np.array(rot_text_polys)
        return data


class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """Random resize the image
        Args:
         input_size:  resize size
         ramdon_rate: random rate
         keep_ratio: if keep the ratio of image
        Returns:
            dict:the processed data
        """
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, list) or isinstance(size, tuple) or isinstance(size, np.ndarray):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """call the class
        
        Args:  
         data: {'image':,'text_polys':,'texts':,'ignore_tags':}
        
        Returns:
            dict:the processed data
        """
        if random.random() > self.random_rate:
            return data
        im = data['image']
        text_polys = data['text_polys']

        if self.keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data['image'] = im
        data['text_polys'] = text_polys
        return data


def resize_image(image, short_size):
    height, width, _ = image.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, (new_width / width, new_height / height)


class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        Args:
         size:  if is list:[w,h]
         
        Returns:
            dict:the processed data
        """
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """rescale the image
        
        Args:
         data: {'image':,'text_polys':,'texts':,'ignore_tags':}
         
        Returns:
            dict:the processed data
        """
        im = data['image']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # keep the short length >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # im, scale = resize_image(im, self.short_size)
            if self.resize_text_polys:
                # text_polys *= scale
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        data['image'] = im
        data['text_polys'] = text_polys
        return data


class HorizontalFlip:
    def __init__(self, random_rate):
        """
         random_rate: random rate
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """horizontal flip the image 
        
        Args:
         data: {'image':,'text_polys':,'texts':,'ignore_tags':}
        
        Returns:
            dict:the processed data
        
        """
        if random.random() > self.random_rate:
            return data
        im = data['image']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        data['image'] = flip_im
        data['text_polys'] = flip_text_polys
        return data


class VerticallFlip:
    def __init__(self, random_rate):
        """
         random_rate: random efficiency
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        
        Args:
         data: {'image':,'text_polys':,'texts':,'ignore_tags':}
        
        Returns:
            dict:the processed data
        """
        if random.random() > self.random_rate:
            return data
        im = data['image']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        data['image'] = flip_im
        data['text_polys'] = flip_text_polys
        
        return data