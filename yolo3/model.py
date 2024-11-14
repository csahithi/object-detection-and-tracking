import tensorflow as tf
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import wraps
import numpy as np
from yolo3.utils import compose

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def yolo_head(feats, anchors, num_classes, input_shape):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Ensure anchors and input_shape are float32 to prevent type mismatches
    anchors_tensor = tf.cast(tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2]), dtype=tf.float32)
    input_shape = tf.cast(input_shape, dtype=tf.float32)

    grid_shape = tf.shape(feats)[1:3]  # height, width
    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, feats.dtype)

    feats = tf.cast(tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]),
                    dtype=tf.float32)

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / input_shape[::-1]
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes for given input and image shape."""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    # Ensure consistent types by converting input_shape and image_shape to float32
    input_shape = tf.cast(tf.convert_to_tensor(input_shape), dtype=tf.float32)
    image_shape = tf.cast(tf.convert_to_tensor(image_shape), dtype=tf.float32)

    # Calculate the new shape with consistent data types
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # Scale boxes back to original image shape
    boxes *= tf.concat([image_shape, image_shape], axis=0)
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(3):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
    boxes_, scores_, classes_ = [], [], []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)
    return boxes_, scores_, classes_
