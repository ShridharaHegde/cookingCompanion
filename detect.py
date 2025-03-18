from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

im = cv2.imread("images/fridge0.png")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused"),
               scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

result = v.get_image()[:, :, ::-1]
cv2.imshow("Detected Objects", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused")
class_ids = outputs["instances"].pred_classes.cpu().numpy()
class_labels = [metadata.thing_classes[i] for i in class_ids]
print("Detected Objects:", class_labels)


