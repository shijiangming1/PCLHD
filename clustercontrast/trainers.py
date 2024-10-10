from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class ClusterContrastTrainer_DCL(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_DCL, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)

    def train(self, epoch, data_loader_ir, data_loader_rgb,
              optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = self._forward(inputs_rgb, inputs_ir,
                                                                                             label_1=labels_rgb,
                                                                                             label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir + loss_rgb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)


class ClusterContrastTrainer_PCLMP(object):
    def __init__(self, encoder, encoder_ema, memory=None):
        super(ClusterContrastTrainer_PCLMP, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory

    def train(self, epoch, data_loader_ir, data_loader_rgb, data_loader_all_ir, data_loader_all_rgb,
              optimizer, print_freq=10, train_iters=400, i2r=None, r2i=None):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = self._forward(inputs_rgb, inputs_ir,
                                                                                             label_1=labels_rgb,
                                                                                             label_2=labels_ir,
                                                                                             modal=0)
            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            # cross contrastive learning
            if r2i:
                rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()
                alternate = True
                if alternate:
                    # accl
                    if epoch % 2 == 1:
                        cross_loss = 1 * self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    else:
                        cross_loss = 1 * self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                else:
                    cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + self.memory_ir(f_out_rgb, rgb2ir_labels.long())
            else:
                cross_loss = torch.tensor(0.0)

            new_loss_rgb = loss_rgb
            new_cross_loss = cross_loss

            
            with torch.no_grad():
                _, f_out_rgb_ema, f_out_ir_ema, labels_rgb_ema, labels_ir_ema, pool_rgb_ema, pool_ir_ema = self._forward_ema(inputs_rgb, inputs_ir,
                                                                                            label_1=labels_rgb,
                                                                                            label_2=labels_ir, modal=0)
            loss_ir_ema = self.memory_ir(f_out_ir_ema, labels_ir_ema, model_name='encoder_ema')
            loss_rgb_ema = self.memory_rgb(f_out_rgb_ema, labels_rgb_ema, model_name='encoder_ema')
            loss_ema = loss_ir_ema + loss_rgb_ema
            loss = loss_ir + new_loss_rgb + 0.25 * new_cross_loss + loss_ema  # total loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()
            # process inputs
            inputs_all_ir, labels_all_ir, indexes_all_ir = self._parse_data_ir(inputs_all_ir)
            inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = self._parse_data_rgb(inputs_all_rgb)
            # forward
            inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)

            _, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir, pool_all_rgb, pool_all_ir = self._forward(
                                                                                        inputs_all_rgb, inputs_all_ir,
                                                                                        label_1=labels_all_rgb,
                                                                                        label_2=labels_all_ir, modal=0)

            loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
            loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)

            loss2 = loss_all_ir + loss_all_rgb

            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            
            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss ema {:.3f}\t'
                      'Loss ir ema {:.3f}\t'
                      'Loss rgb ema {:.3f}\t'
                      'Loss all {:.3f}\t'
                      'Loss all ir {:.3f}\t'
                      'Loss all rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb, loss_ema, loss_ir_ema, loss_rgb_ema, loss2, loss_all_ir, loss_all_rgb))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)
    
    def _forward_ema(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder_ema(x1, x2, modal=modal, label_1=label_1, label_2=label_2)   

    def _update_ema_variables(self, model, ema_model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)