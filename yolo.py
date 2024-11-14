import colorsys
import os
import random
import tensorflow as tf
from timeit import default_timer as timer
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="./test_video/det_t1_video_00315_test.avi")
ap.add_argument("-c", "--class", help="name of class", default="person")
args = vars(ap.parse_args())

class YOLO(object):
    def __init__(self):
        self.model_path = './model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'

        self.score = 0.6
        self.iou = 0.6
        self.model_image_size = (416, 416)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.model_image_size = (416, 416) if args["class"] in ["person", "car", "bicycle", "motorcycle"] else (None, None)
        self.yolo_model = load_model(self.model_path, compile=False)
        print(f'{self.model_path} model, anchors, and classes loaded.')

    def _get_class(self):
        with open(self.classes_path) as f:
            return [c.strip() for c in f.readlines()]

    def _get_anchors(self):
        with open(self.anchors_path) as f:
            anchors = [float(x) for x in f.readline().split(',')]
        return np.array(anchors).reshape(-1, 2)

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32') / 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension

        model_output = self.yolo_model(image_data, training=False)
        out_boxes, out_scores, out_classes = yolo_eval(
            model_output,
            self.anchors,
            len(self.class_names),
            [image.size[1], image.size[0]],  # Use explicit image shape
            score_threshold=self.score,
            iou_threshold=self.iou
        )

        return_boxes = []
        return_class_names = []
        for i, c in reversed(list(enumerate(out_classes.numpy()))):
            predicted_class = self.class_names[c]
            if predicted_class != args["class"]:
                continue
            box = out_boxes[i].numpy()
            return_boxes.append([int(box[1]), int(box[0]), int(box[3] - box[1]), int(box[2] - box[0])])
            return_class_names.append(predicted_class)

        return return_boxes, return_class_names

    def close_session(self):
        tf.keras.backend.clear_session()
