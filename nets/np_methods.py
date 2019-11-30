# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Additional Numpy methods. Big mess of many things!
"""
import numpy as np


# =========================================================================== #
# Numpy implementations of SSD boxes functions.
# =========================================================================== #
def ssd_bboxes_decode(feat_localizations,
                      anchor_bboxes,
                      prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    # Reshape for easier broadcasting.
    l_shape = feat_localizations.shape
    feat_localizations = np.reshape(feat_localizations,
                                    (-1, l_shape[-2], l_shape[-1]))
    yref, xref, href, wref = anchor_bboxes
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])

    # Compute center, height and width
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(feat_localizations)
    bboxes[:, :, 0] = cy - h / 2.
    bboxes[:, :, 1] = cx - w / 2.
    bboxes[:, :, 2] = cy + h / 2.
    bboxes[:, :, 3] = cx + w / 2.
    # Back to original shape.
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes


def ssd_bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            anchors_layer,
                            select_threshold=0.5,
                            img_shape=(300, 300),
                            num_classes=21,
                            decode=True):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # First decode localizations features if necessary.
    if decode:
        localizations_layer = ssd_bboxes_decode(localizations_layer, anchors_layer)

    # Reshape features to: Batches x N x N_labels | 4.
    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = np.argmax(predictions_layer, axis=2)
        scores = np.amax(predictions_layer, axis=2)
        mask = (classes > 0)
        classes = classes[mask]
        scores = scores[mask]
        bboxes = localizations_layer[mask]
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        idxes = np.where(sub_predictions > select_threshold)
        classes = idxes[-1] + 1
        scores = sub_predictions[idxes]
        bboxes = localizations_layer[idxes[:-1]]

    return classes, scores, bboxes


def ssd_bboxes_select(predictions_net,
                      localizations_net,
                      anchors_net,
                      select_threshold=0.5,
                      img_shape=(300, 300),
                      num_classes=21,
                      decode=True):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    # l_layers = []
    # l_idxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            select_threshold, img_shape, num_classes, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
        # Debug information.
        # l_layers.append(i)
        # l_idxes.append((i, idxes))

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes


# =========================================================================== #
# Common functions for bboxes handling and selection.
# =========================================================================== #
def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes


def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes


def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def bboxes_intersection(bboxes_ref, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score


def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size - 1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i + 1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i + 1):] != classes[i])
            keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


def patches_intersect(bboxes_ref, bboxes2):
    a, b = bboxes_ref, bboxes2
    x1 = max(a[1], b[1])
    y1 = max(a[0], b[0])
    x2 = min(a[3], b[3])
    y2 = min(a[2], b[2])
    if x1 < x2 and y1 < y2:
        return np.array([y1, x1, y2, x2])
    else:
        return None


def bboxes_nms_fast(classes, scores, bboxes, threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    pass


def patches_intersect(bboxes_ref, bboxes2):
    a, b = bboxes_ref, bboxes2
    x1 = max(a[1], b[1])
    y1 = max(a[0], b[0])
    x2 = min(a[3], b[3])
    y2 = min(a[2], b[2])
    if x1 < x2 and y1 < y2:
        return np.array([y1, x1, y2, x2])
    else:
        return None


def bboxes_intersect(bboxes_ref, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    intersect_bboxes = np.zeros_like(bboxes2)

    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)

    # Intersection bbox and volume.
    intersect_bboxes[:, 0] = np.maximum(bboxes_ref[0], bboxes2[0])
    intersect_bboxes[:, 1] = np.maximum(bboxes_ref[1], bboxes2[1])
    intersect_bboxes[:, 2] = np.minimum(bboxes_ref[2], bboxes2[2])
    intersect_bboxes[:, 3] = np.minimum(bboxes_ref[3], bboxes2[3])

    return intersect_bboxes


def intersect_with_patches(bboxes_ref, patches):
    bboxes_intersects = bboxes_intersect(bboxes_ref, patches)
    int_ymin = bboxes_intersects[:, 0]
    int_xmin = bboxes_intersects[:, 1]
    int_ymax = bboxes_intersects[:, 2]
    int_xmax = bboxes_intersects[:, 3]

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w

    return int_vol > 0


def splitted_bboxes_nms(classes, scores, bboxes, patches, nms_threshold):
    """Apply non-maximum selection to bounding boxes.
        And bounding boxes merging.
    """
    print(len(classes))
    patch_intersections = []
    for i, _ in enumerate(patches):
        if i < len(patches) - 1:
            intersect = patches_intersect(patches[i], patches[i + 1])
            if intersect is not None:
                patch_intersections.append(intersect)

    patch_intersections = np.array(patch_intersections)
    int_with_patches = [np.any(intersect_with_patches(box, patch_intersections)) for box in bboxes]

    keep_bboxes = np.ones_like(scores, dtype=np.bool)

    int_idx = np.nonzero(int_with_patches)[0]
    if len(int_idx) > 1:
        for idx, i in enumerate(int_idx):
            if i < len(keep_bboxes) - 1:
                # Computer overlap with bboxes which are following.
                overlap = bboxes_jaccard(bboxes[i], bboxes[int_idx[idx + 1:]])
                # Overlap threshold for keeping + checking part of the same class
                print(overlap)
                keep_overlap = np.logical_or(overlap < nms_threshold, classes[int_idx[idx + 1:]] != classes[i])
                keep_bboxes[int_idx[idx + 1:]] = np.logical_and(keep_bboxes[int_idx[idx + 1:]], keep_overlap)

    idxes = np.where(keep_bboxes)
    print(len(classes[idxes]))

    return classes[idxes], scores[idxes], bboxes[idxes]


def bboxes_calibrate(bboxes, input_shape, patch_shape, patch_starting_points):
    '''
    boxes : [[y_upper, x_left, y_lower, x_right], [ ... ], ..., [ ... ]]
    img : 3-d np array
    patch_arr_coord : [y_upper, x_left, y_lower, x_right]; coordination type of matplot library.
                      the positions of xys are mutually exchanged.

    '''
    bboxes[:, 0] = calibrate(bboxes[:, 0], input_shape[0], patch_shape[0], patch_starting_points[0])
    bboxes[:, 1] = calibrate(bboxes[:, 1], input_shape[1], patch_shape[1], patch_starting_points[1])
    bboxes[:, 2] = calibrate(bboxes[:, 2], input_shape[0], patch_shape[0], patch_starting_points[0])
    bboxes[:, 3] = calibrate(bboxes[:, 3], input_shape[1], patch_shape[1], patch_starting_points[1])

    return bboxes


def calibrate(bbox_corner, input_size, patch_size, patch_start_point):
    return (bbox_corner * patch_size + patch_start_point) / input_size


def resize_test_image(np_image, input_shape):
    from scipy.misc import imresize

    np_image = imresize(np_image, input_shape)
    return np_image


def slice_input_image_into_patches(preprocessed_input, patch_shape):
    input_shape_ = preprocessed_input.shape
    cropping_target_boundaries = [0, 0, input_shape_[0], input_shape_[1]]
    cropping_regions = get_cropping_regions_for_patches_numpy(cropping_target_boundaries, patch_shape)
    input_patches_ = []
    for _, cr in enumerate(cropping_regions):
        input_patches_.append(slice_image(preprocessed_input, cr))

    return input_patches_


def slice_image(np_image, cropping_regions):
    cr = cropping_regions
    return np_image[cr[0]:cr[2], cr[1]:cr[3], :]


def get_cropping_regions_for_patches_numpy(cropping_target_boundaries, patch_shape):
    num_patch = get_number_of_patches_for_sub_region_numpy(cropping_target_boundaries, patch_shape)
    vertical_num_patch, horizontal_num_patch = num_patch

    upper_corner, left_corner, lower_corner, right_corner = cropping_target_boundaries
    target_horizontal_boundary = [left_corner, right_corner]
    target_vertical_boundary = [upper_corner, lower_corner]

    patch_horizontal_boundaries = get_patch_boundary_points_numpy(horizontal_num_patch,
                                                                  target_horizontal_boundary,
                                                                  patch_shape[0],
                                                                  axis='x')
    patch_vertical_boundaries = get_patch_boundary_points_numpy(vertical_num_patch,
                                                                target_vertical_boundary,
                                                                patch_shape[1],
                                                                axis='y')

    cropping_regions = []
    '''
    row-wise patch cropping
    (upper_corner, left_corner, lower_corner, right_corner)
    ex) (0, 0, 3, 3) => (0, 2, 3, 5) => (2, 0, 5, 3) => (2, 2, 5, 5)
    '''
    for _, h_upper_boundary in enumerate(patch_vertical_boundaries):
        for _, v_left_boundary in enumerate(patch_horizontal_boundaries):
            cropping_region = [
                int(h_upper_boundary),
                int(v_left_boundary),
                int(h_upper_boundary + patch_shape[0]),
                int(v_left_boundary + patch_shape[1])
            ]
            cropping_regions.append(cropping_region)

    return cropping_regions


def get_patch_boundary_points_numpy(num_patch, cropping_target_boundary, patch_size, axis):
    target_min_boundary, target_max_boundary = cropping_target_boundary
    boundary_distance = target_max_boundary - target_min_boundary
    min_num_patch = np.max([np.round(boundary_distance / patch_size), 1])
    # assert num_patch >= min_num_patch, "at least %d patches" % min_num_patch

    if num_patch == 1:
        over_lap_size = 0.
        patch_min_boundaries = np.arange(target_min_boundary,
                                         target_max_boundary,
                                         patch_size - over_lap_size)[:int(num_patch)]
    else:
        if axis == 'x':
            over_lap_size = get_overlap_size(num_patch, patch_size, boundary_distance)
            patch_min_boundaries = np.arange(target_min_boundary,
                                             target_max_boundary,
                                             patch_size - over_lap_size)[:int(num_patch)]
        elif axis == 'y':
            over_lap_size = get_overlap_size(num_patch, patch_size, boundary_distance)
            patch_min_boundaries = np.arange(target_min_boundary,
                                             target_max_boundary,
                                             patch_size - over_lap_size)[:int(num_patch)]
        else:
            raise Exception('Axis must be one of "x" or "y".')

    return patch_min_boundaries


def get_overlap_size(num_patch, patch_size, boundary_distance):
    # overlap_size must be integer, since it will used for slice an image.
    if patch_size * num_patch - boundary_distance > 0:
        return np.round((patch_size * num_patch - boundary_distance) / (num_patch - 1))
    else:
        return 0


def get_number_of_patches_for_sub_region_numpy(target_cropping_target_boundaries, patch_shape):
    upper_coner, left_coner, lower_corner, right_corner = target_cropping_target_boundaries
    img_res = [(lower_corner - upper_coner), (right_corner - left_coner)]
    num_patch = int(np.max([np.round(img_res[0] / patch_shape[0]), 1])), int(np.ceil(img_res[1] / patch_shape[1]))
    # num_patch = int(np.ceil(img_res[0] / patch_shape[0])), int(np.ceil(img_res[1] / patch_shape[1]))

    return num_patch


def cut_tensor_by_thresholding(classes, scores, bboxes, filter_threshold=0.75):
    mask = scores[0] > filter_threshold
    head_classes = (classes[0][mask],)
    head_scores = (scores[0][mask],)
    head_bboxes = (bboxes[0][mask],)

    imask = np.logical_not(mask)
    patch_classes = classes[1:] + [classes[0][imask]]
    patch_scores = scores[1:] + [scores[0][imask]]
    patch_bboxes = bboxes[1:] + [bboxes[0][imask]]

    return (head_classes, head_scores, head_bboxes), (patch_classes, patch_scores, patch_bboxes)


def flatten_net_outs(classes, scores, bboxes):
    return np.concatenate(classes, axis=0), \
           np.concatenate(scores, axis=0), \
           np.concatenate(bboxes, axis=0)


def get_normalized_patch_areas(input_shape, crop_areas):
    return [get_normalized_patch_area(input_shape, crop_area) for crop_area in crop_areas]


def get_normalized_patch_area(input_shape, crop_area):
    patch = np.array(crop_area, dtype=np.float32)
    patch[0::2] = patch[0::2] / input_shape[0]
    patch[1::2] = patch[1::2] / input_shape[1]
    return np.array(patch)
