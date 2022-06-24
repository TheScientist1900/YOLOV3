from yacs.config import CfgNode as CN
from .utils import get_classes, get_anchors
import os

_C = CN()

_C.num_workers = 0
_C.pin_memory = True


_C.root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

_C.arch = 'mobilenetv2'
_C.pretrained = False
_C.checkpoint = ''
_C.data_name = 'VOC2007'
_C.resume = False
_C.out_dir = 'exp'

_C.cuda = True
_C.gpus = [0]

_C.data = CN()
_C.data.data_name = 'VOC2007'
_C.data.root = os.path.join(_C.root, 'VOCdevkit', _C.data_name)
_C.data.class_names = get_classes(os.path.join(_C.data.root, 'voc_classes.txt'))[0]
_C.data.num_classes = get_classes(os.path.join(_C.data.root, 'voc_classes.txt'))[1]
_C.data.input_shape = [416, 416]

_C.model = CN()
_C.model.num_box = 3
_C.model.anchors = get_anchors(os.path.join(_C.data.root, 'yolo_anchors.txt'))[0]
_C.model.anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]

_C.loss = CN()
_C.loss.ignore_thresh = 0.5
_C.loss.loc_ratio = 0.05
_C.loss.cls_ratio = 1 * (_C.data.num_classes / 80)
_C.loss.conf_ratio = 5 * (_C.data.input_shape[0] * _C.data.input_shape[1]) / (416 ** 2)
_C.loss.balance = [0.4, 1.0, 4]

_C.train = CN()
_C.train.freeze_train = False
_C.train.freeze_epoch = 50
_C.train.freeze_batch_size = 8

_C.train.freeze_lr = 3e-4
_C.train.unfreeze_epoch = 100
_C.train.unfreeze_lr = 1e-3
_C.train.unfreeze_batch_size = 2
_C.train.shuffle = False

_C.predict = CN()
_C.predict.checkpoint = os.path.join(_C.root, r'model_data\yolo_weights.pth')
_C.predict.cuda = True
_C.predict.conf_thresh = 0.5
_C.predict.nms_thresh = 0.3
_C.predict.video_path = os.path.join(_C.root, r'video\raw\video1.mp4')
_C.predict.video_save_path = os.path.join(_C.root, r'video\out')
_C.predict.video_fps = 25.0

_C.map = CN()
_C.map.conf_thresh = 0.001
_C.map.nms_thresh = 0.5
_C.map.cuda = True
_C.map.map_out_path = os.path.join(_C.root, 'map_out')
_C.map.checkpoint = os.path.join(_C.root, r'model_data\yolo_weights.pth')
_C.map.cpk_dir = r'2022-05-01-09-46-37'



def update_cfg(cfg, args):
    cfg.defrost()
    cfg.data.name = args.data_name
    cfg.data.root = os.path.join(cfg.root, 'VOCdevkit', cfg.data_name)
    arg_list = []
    for k, v in vars(args).items():
        arg_list.append(k)
        arg_list.append(v)
    
    cfg.merge_from_list(arg_list)
    cfg.freeze()
    return cfg