import os
import io
import cv2
import numpy as np
import logging
from PIL import Image
import json
import torch, torchvision
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


logger = logging.getLogger(__name__)

# Sets of parameters. Please change this part accordingly if you are using another model
MODEL_ZOO_CONFIG_FILE = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
INPUT_MAX_SIZE_TEST = 1600
MODEL_WEIGHTS = "model_best.pth"
ROI_HEADS_SCORE_THRESH_TEST = 0.9
ROI_HEADS_BATCH_SIZE_PER_IMAGE = 512
ROI_HEADS_NUM_CLASSES = 1 
RETINANET_NUM_CLASSES = 1
CONFIDENCE_THRESHOLD = 0.12
KDE_KERNEL_BANDWIDTH = 0.05


def model_fn(model_dir):
    """
    Load the model from the directory model_dir and returns a deserialized Pytorch model
    """
    logger.info("Loading model...")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_CONFIG_FILE))
    cfg.INPUT.MAX_SIZE_TEST = INPUT_MAX_SIZE_TEST
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, MODEL_WEIGHTS)  # path to the model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ROI_HEADS_SCORE_THRESH_TEST   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = ROI_HEADS_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = ROI_HEADS_NUM_CLASSES  
    cfg.MODEL.RETINANET.NUM_CLASSES = RETINANET_NUM_CLASSES
#     cfg.MODEL.DEVICE='cpu'
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model


def _resize_img(img, target_height = INPUT_MAX_SIZE_TEST, target_width = INPUT_MAX_SIZE_TEST):
    """
    _resize_img resizes the image specified by the parameter img to the specified target width and height
    by maintaining the aspect ratio and adding a gray padding (127, 127, 127).

    :param target_height: target height.
    :param target_width: target width.
    
    :return: border_image: resized image
    :return: params: transformation parameters which were applied to img
    """ 
    
    h, w, _ = img.shape
    size_percent = min(target_height/h, target_width/w)
    new_h = int(h * size_percent)
    new_w = int(w * size_percent)
    new_img = cv2.resize(img, dsize = (new_w, new_h), interpolation = cv2.INTER_NEAREST)
    
    color = [127, 127, 127]
    border_img = np.full((target_height, target_width, 3), color, dtype = np.uint8)
    h, w, _ = new_img.shape
    
    h_margin = (target_height - h) // 2
    w_margin = (target_width - w) // 2
    
    params = [size_percent, h_margin, w_margin]
    border_img[h_margin:h_margin + h, w_margin:w_margin + w] = new_img
    
    return border_img, params


def input_fn(request_body, request_content_type='application/x-image'):
    """
    Take request data and deserializes the data into an object for prediction.
    """
    logger.info("Loading input data...")

    if request_content_type == 'application/x-image':
        logger.info(request_body)
        logger.info(io.BytesIO(request_body))
        logger.info(Image.open(io.BytesIO(request_body)))
        original_image = np.array(Image.open(io.BytesIO(request_body)).convert("RGB"))
        original_image = original_image[:, :, ::-1]  # RGB to BGR
        original_image, _ = _resize_img(original_image)
        height, width = original_image.shape[:2]
        aug = T.ResizeShortestEdge([800, 800], INPUT_MAX_SIZE_TEST)
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return inputs

    raise Exception('Requested unsupported ContentType in content_type: ' + request_content_type)


def predict_fn(input_data, model):
    """
    Take the deserialized request object and performs inference against the loaded model
    """
    logger.info("Generating prediction...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(input_data)
    model = model.to(device)
    height, width = input_data['height'], input_data['width']
    input_data["image"] = input_data["image"].to(device)
    logger.info(input_data["image"].shape)
    model.eval()

    with torch.no_grad():
        logger.info("Starting forward propagation...")
        outputs = model([input_data])[0]
        logger.info(outputs)
        data = outputs["instances"].to("cpu")
        scores = data.get("scores").numpy()
        bboxes = data.get("pred_boxes").tensor.numpy()

        pred_data = {}
        logger.info(scores)
        pred_data["scores"] = scores.tolist()
        pred_data["rois"] = bboxes.tolist()
        return pred_data


def output_fn(prediction, content_type='application/json'):
    """
    Serialize the output of the prediction
    """
    logger.info("Serializing the generated output...")
    logger.info(prediction)

    confidence_threshold = CONFIDENCE_THRESHOLD
    kde_kernel_bandwidth = KDE_KERNEL_BANDWIDTH

    scores = prediction["scores"]
    if len(scores) == 0:
        clusters = []
    else:
        a = np.array(scores).reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=kde_kernel_bandwidth).fit(a)
        s = np.arange(0, 1, 0.001)
        e = kde.score_samples(s.reshape(-1,1))
        mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
        if len(mi) == 0:
            clusters = [scores]
        else:
            clusters = []
            clusters.append(a[a < s[mi][0]].tolist())
            for idx in range(0,len(mi)-1):
                clusters.append(a[(a >= s[mi][idx]) * (a <= s[mi][idx+1])].tolist())
            clusters.append(a[a >= s[mi][len(mi)-1]].tolist())

    bboxes = prediction["rois"]
    pred_bboxes = []
    pred_scores = []
    for i in range(len(scores)):
        if scores[i] not in clusters[-1]:
            continue
        if scores[i] >= confidence_threshold:
            pred_bboxes.append(bboxes[i])
            pred_scores.append(scores[i])

    response = {}
    if len(pred_scores) == 0:
        response['signature'] = {'score': [], 'roi': []}
    else:
        for i in range(len(pred_scores)):
            response[f'signature{i}'] = {'score': pred_scores[i], 'roi': pred_bboxes[i]}
    
    if content_type == 'application/json':
#         return json.dumps({'scores': pred_scores, 'rois': pred_bboxes}), content_type
        return json.dumps(response)
    
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
