import time
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.utils import get_lr


class YOLOV3Trainer():
    def __init__(self, cfg, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.cuda = cfg.cuda

    def train_epoch(self, loss, train_loader, train_step, epoch, epoch_num, logger):
        total_loss, loc_loss, cls_loss, conf_loss = 0, 0, 0, 0

        print('start train')
        self.model.train()
        with tqdm(desc='{}/{}'.format(epoch, epoch_num), total=train_step, mininterval=0.3, postfix=dict) as pbar:
            for iter, batch in enumerate(train_loader):
                with torch.no_grad():
                    if self.cuda:
                        images = torch.from_numpy(batch[0]).type(torch.FloatTensor).cuda()
                        targets = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in batch[1]]
                    else:
                        images = torch.from_numpy(batch[0]).type(torch.FloatTensor)
                        targets = [torch.from_numpy(i).type(torch.FloatTensor) for i in batch[1]]

                self.optimizer.zero_grad()

                preds = self.model(images)
                loss_all_layer = [0, 0, 0, 0]
                for l in range(len(preds)):
                    
                    losses = loss(l, preds[l], targets)
                    loss_all_layer = [losses[i] + loss_all_layer[i] for i in range(4)]

                loss_value = loss_all_layer[0]

                loss_value.backward()
                self.optimizer.step()

                total_loss += loss_all_layer[0].item()
                conf_loss += loss_all_layer[1].item()
                loc_loss += loss_all_layer[2].item()
                cls_loss += loss_all_layer[3].item()

                pbar.set_postfix(**{
                    'total': total_loss / (iter + 1),
                    'conf': conf_loss / (iter + 1),
                    'loc': loc_loss / (iter + 1),
                    'cls': cls_loss / (iter + 1),
                    'lr' : get_lr(self.optimizer),
                    'time': time.strftime("%H:%M", time.localtime()) 
                    })
                pbar.update(1)
        logger.info(
            "Epoch: {}\nTrain\n\ttotoal_loss: {:.5f}\n\tconf_loss: {:.5f}\n\tloc_loss: {:.5f}\n\tcls_loss: {:.5f}".format(
                    epoch, 
                    total_loss / train_step, 
                    conf_loss / train_step, 
                    loc_loss / train_step, 
                    cls_loss / train_step, 
                ))
        return total_loss / train_step    
    def validate(self, loss, val_loader, val_step, epoch, epoch_num, logger):
        total_loss, loc_loss, cls_loss, conf_loss = 0, 0, 0, 0
        
        self.model.eval()
        print('start validate')
        
        with tqdm(desc='{}/{}'.format(epoch, epoch_num), total=val_step, mininterval=0.3, postfix=dict) as pbar:
            for iter, batch in enumerate(val_loader):
                with torch.no_grad():
                    if self.cuda:
                        images = torch.from_numpy(batch[0]).type(torch.FloatTensor).cuda()
                        targets = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in batch[1]]
                    else:
                        images = torch.from_numpy(batch[0]).type(torch.FloatTensor)
                        targets = [torch.from_numpy(i).type(torch.FloatTensor) for i in batch[1]]
                    self.optimizer.zero_grad()

                    preds = self.model(images)
                    loss_all_layer = [0, 0, 0, 0]
                    for l in range(len(preds)):
                        
                        losses = loss(l, preds[l], targets)
                        loss_all_layer = [losses[i] + loss_all_layer[i] for i in range(4)]

                    total_loss += loss_all_layer[0].item()
                    conf_loss += loss_all_layer[1].item()
                    loc_loss += loss_all_layer[2].item()
                    cls_loss += loss_all_layer[3].item()
                        
                    pbar.set_postfix(**{
                        'total': total_loss / (iter + 1),
                        'conf': conf_loss / (iter + 1),
                        'loc': loc_loss / (iter + 1),
                        'cls': cls_loss / (iter + 1),
                        'lr' : get_lr(self.optimizer),
                        'time': time.strftime("%H:%M", time.localtime()) 
                        })
                    pbar.update(1)
        logger.info(
            "Epoch: {}\nValidate\n\ttotoal_loss: {:.5f}\n\tconf_loss: {:.5f}\n\tloc_loss: {:.5f}\n\tcls_loss: {:.5f}".format(
                    epoch, 
                    total_loss / val_step, 
                    conf_loss / val_step, 
                    loc_loss / val_step, 
                    cls_loss / val_step, 
                ))
        return total_loss / val_step