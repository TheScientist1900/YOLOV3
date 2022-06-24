import argparse
import os
from loss.yolov3_loss import YOLOV3Loss
from nets.yolov3_trainer import YOLOV3Trainer
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from torch.utils.data.dataloader import DataLoader
import torch
from utils.config import _C as cfg
from utils.config import update_cfg
from utils.utils import create_logger, get_model_summary, save_checkpoint
from nets.yolov3 import YOLOV3
import numpy as np
import torch.nn as nn
def parse_arg():
    parser = argparse.ArgumentParser('Yolov3 training')
    parser.add_argument('--data_name', default='VOC2007',help='the dir name under the VOCdevkit')
    parser.add_argument('--arch', default='mobilenetv2',help='')
    parser.add_argument('--pretrained', default=True, help='if load the backbone weights')
    parser.add_argument('--checkpoint', default='', help='the path of the cpk file')
    parser.add_argument('--resume', default=False, help='if continue to train from cpk file')
    return parser.parse_args()
def main(cfg):
    args = parse_arg()
    cfg = update_cfg(cfg, args)

    final_output_dir, logger = create_logger(cfg)
    logger.info(cfg)

    begin_epoch = 0
    best_perf = 100
    is_best = False

    freeze = cfg.train.freeze_train
    unfreeze_flag = False
    model = YOLOV3(cfg)
    if freeze:
        batch_size = cfg.train.freeze_batch_size
        lr = cfg.train.freeze_lr
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        batch_size = cfg.train.unfreeze_batch_size
        lr = cfg.train.unfreeze_lr
        unfreeze_flag = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    loss_history = LossHistory(final_output_dir, )
    
    if cfg.cuda:
        model = nn.DataParallel(model, device_ids = cfg.gpus)
        model = model.cuda()

    if cfg.checkpoint != '':
        cpk = os.path.join(cfg.root, cfg.checkpoint)
        logger.info('=> loading weights from {}'.format(cpk))
        model_dict = model.state_dict()
        cpk_dict = torch.load(cpk)
        pretrained_dict =  cpk_dict if 'state_dict' not in cpk_dict.keys() else cpk_dict['state_dict']
        # pretrained_dict = {k:v for k, v in pretrained_dict.items() if np.shape(v) == np.shape(model_dict[k])}
        new_pretrained_dict = {}
        for k in list(pretrained_dict.keys()):
            v = pretrained_dict[k]
            if not k.__contains__('module'):
                k = 'module.' + k
            
            if np.shape(v) == np.shape(model_dict[k]):
                new_pretrained_dict[k] = v
            
        model_dict.update(new_pretrained_dict)
        model.load_state_dict(model_dict)

        if cfg.resume:
            optimizer.load_state_dict(cpk_dict['optimizer'])
            lr_scheduler.load_state_dict(cpk_dict['lr_scheduler'])
            begin_epoch = cpk_dict['epoch'] + 1
            best_perf = cpk_dict['best_perf']
            loss_history.load_loss(os.path.join(cpk[:-15], 'loss'))
            
            if begin_epoch >= cfg.train.freeze_epoch:
                freeze = False
                batch_size = cfg.train.unfreeze_batch_size
                unfreeze_flag = True
                for param in model.module.backbone.parameters():
                    param.requires_grad = True
            logger.info('=> continue to train from epoch {}'.format(begin_epoch))

    trainer = YOLOV3Trainer(cfg, model, optimizer)
    loss = YOLOV3Loss(cfg)
    
    train_dataset = YoloDataset(cfg, mode='train',)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=cfg.train.shuffle,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
        collate_fn=yolo_dataset_collate)
    train_step = train_dataset.__len__() // batch_size

    val_dataset = YoloDataset(cfg, mode='val', )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
        collate_fn=yolo_dataset_collate)
    val_step = val_dataset.__len__() // batch_size

    if freeze:
        logger.info("=> start freeze training")
    else:
        logger.info("=> start unfreeze training")

    dump_input = torch.rand(
        (1, 3, cfg.data.input_shape[0], cfg.data.input_shape[1])
    )
    logger.info(get_model_summary(model, dump_input))

    for epoch in range(begin_epoch, cfg.train.unfreeze_epoch):
        if epoch >= cfg.train.freeze_epoch and not unfreeze_flag and cfg.train.freeze:
            logger.info('=> start unfreeze training')
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.unfreeze_lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
            for param in model.module.backbone.parameters():
                param.requires_grad = True
            trainer = YOLOV3Trainer(cfg, model, optimizer)

            batch_size = cfg.train.unfreeze_batch_size
            train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=cfg.train.shuffle,
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
                collate_fn=yolo_dataset_collate)
            train_step = train_dataset.__len__() // batch_size
            
            val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
                collate_fn=yolo_dataset_collate)
            val_step = val_dataset.__len__() // batch_size

            unfreeze_flag = True

        train_loss = trainer.train_epoch(loss, train_loader, train_step, epoch, 
                    cfg.train.unfreeze_epoch, logger)
        
        val_loss = trainer.validate(loss, val_loader, val_step, epoch, cfg.train.unfreeze_epoch, logger)
        loss_history.append_loss(train_loss, val_loss)

        logger.info("=> saving checkpoint to {}".format(final_output_dir))
        if val_loss < best_perf:
            is_best = True
            best_perf = val_loss
            logger.info("=> saving model_best.pth file to {}".format(final_output_dir))
        
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_perf': best_perf
        }, is_best, final_output_dir)

        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        # torch.save(model.module.state_dict(), final_model_state_file)
        pytorch_v = torch.__version__
        if pytorch_v >= '1.6.0':
            torch.save(model.module.state_dict(), final_model_state_file, 
                _use_new_zipfile_serialization=False)
        else:
            torch.save(model.module.state_dict(), final_model_state_file,)

        lr_scheduler.step()
        is_best = False
if __name__ == '__main__':
    main(cfg)