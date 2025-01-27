import cv2
from numpy.core.fromnumeric import clip
from numpy.lib.arraysetops import isin
import torch
import numpy as np
from shapely.geometry import Polygon
import pyclipper


class ToyDetDecoder(object):

    thresh = 0.6
    box_thresh = 0.6
    max_candidates = 100
    dest = 'binary'

    def __init__(self, thresh=0.5, box_thresh=0.70, **kwargs):
        self.min_size = 1
        self.scale_ratio = 0.4

        self.thresh = thresh

        self.box_thresh = box_thresh

    def decode(self, fmap, width, height, is_output_polygon=False):
        '''
        fmap: feature map  Tensor (1,1,H,W)
        is_out_polygon: bool 
        '''

        segmentation = self.binarize(fmap)
        boxes, scores = self.boxes_from_bitmap(segmentation, width, height)

        return boxes, scores

    def decode_batch(self, scale_xys, heats, down_scale=4):

        bboxes_list = []
        scores_list = []
        batch = heats.size()[0]

        for i in range(batch):
            scale_x, scale_y = scale_xys[i]
            bboxes, scores = self.boxes_from_bitmap(
                torch.squeeze(heats[i, :, :, :]))
            if not isinstance(bboxes, list):
                bboxes[:, 0::2] = bboxes[:, 0::2] * \
                    scale_x * down_scale
                bboxes[:, 1::2] = bboxes[:, 1::2] * \
                    scale_y * down_scale

            bboxes_list.append(bboxes)
            scores_list.append(scores)

        return bboxes_list, scores_list

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap=None, max_candidates=100):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        # assert _bitmap.size(0) == 1
        # bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        if not _bitmap:
            _bitmap = pred

        bitmap = _bitmap.cpu().detach().numpy()
        clip_h, clip_w = bitmap.shape

        pred = pred.cpu().detach().numpy()
        # cv2.imshow("bitmap",bitmap*255)
        height, width = bitmap.shape
        bitmap[np.where(bitmap>1.0)]=1.0
        bitmap[np.where(bitmap<0.0)]=0.0
        
        
        
        _, mask = cv2.threshold(
            (bitmap * 255).astype(np.uint8), self.box_thresh*255, 255, cv2.THRESH_BINARY)
        # cv2.imshow("mask",mask)
        
        
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        # cv2.imshow("mask2",mask)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), max_candidates)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)

            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            # if self.box_thresh > score:
            #     continue
            if score < 0.001:
                continue
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                 continue
            box = np.array(box)

            box[:, 0] = np.clip(box[:, 0], 0, clip_w)
            box[:, 1] = np.clip(box[:, 1], 0, clip_h)

            boxes.append(box)

            scores.append(score)

        if boxes:
            boxes = np.stack(boxes, 0)
            scores = np.stack(scores, 0)

        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour, shrink=True):
        try:
            rect = cv2.minAreaRect(contour)
            # if shrink:
            #     rect = (rect[0], (rect[1][0], rect[1][0]*0.9), rect[2])
            points = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])

            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if points[1][1] > points[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if points[3][1] > points[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2

            box = [points[index_1], points[index_2],
                points[index_3], points[index_4]]
        
            return box, min(rect[1])
        except Exception as e:
            print(e)
            print(contour)
            return [],-1

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
