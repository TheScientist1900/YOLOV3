import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
# from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir):
        self.log_dir    = os.path.join(log_dir, "loss")
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        
    def append_loss(self, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def load_loss(self, loss_dir):
        losses = open(os.path.join(loss_dir, 'epoch_loss.txt'), 'r', encoding='utf-8').readlines()
        for l in losses:
          self.losses.append(float(l))
        
        val_loss = open(os.path.join(loss_dir, 'epoch_val_loss.txt'), 'r', encoding='utf-8').readlines()
        for l in val_loss:
          self.val_loss.append(float(l))
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            for l in self.losses:
                f.write(str(l))
                f.write("\n")
            f.close()
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            for l in self.val_loss:
                f.write(str(l))
                f.write("\n")
            f.close()
        self.loss_plot()