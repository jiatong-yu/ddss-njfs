import yolov5

def _load_yolo5_model():
    """
    Returns:
        Yolo5 model. see https://github.com/fcakyon/yolov5-pip for documentation
    """
    model = yolov5.load('yolov5s.pt')
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 20  # maximum number of detections per image
    return model


yolo = _load_yolo5_model()