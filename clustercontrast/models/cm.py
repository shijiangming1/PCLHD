import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from .losses import CrossEntropyLabelSmooth, FocalTopLoss
from IPython import embed


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()
            
            hard = np.argmin(np.array(distances))
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * features[hard]
            ctx.features[index+nums] /= ctx.features[index+nums].norm()


        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, num_instances):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features) // ctx.num_instances 
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        # 更新实例特征
        for index, features in batch_centers.items():
            indexes = [index + nums * i for i in range(0, ctx.num_instances)]
            if len(features) >= 16:
                ids = np.random.choice(len(features), len(indexes), replace=False)
            else:
                ids = np.random.choice(len(features), len(indexes), replace=True)

            ctx.features[indexes] = torch.stack(features)[ids]
            # ctx.features[indexes] /= ctx.features[indexes].norm()
            
        return grad_inputs, None, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5, num_instances=16):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), num_instances)


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mode='CM', hard_weight=0.5, smooth=0,
                 num_instances=16):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode

        if smooth > 0:
            self.cross_entropy = CrossEntropyLabelSmooth(self.num_samples, 0.1, True)
            print('>>> Using CrossEntropy with Label Smoothing.')
        else:
            self.cross_entropy = nn.CrossEntropyLoss().cuda()

        if self.cm_type == 'CM':
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        elif self.cm_type == 'CMhybrid':
            self.register_buffer('features', torch.zeros(2 * num_samples, num_features))
        elif self.cm_type == 'CMhard':
            self.hard_weight = hard_weight
            self.num_instances = num_instances
            self.register_buffer('features', torch.zeros(2 * num_samples, num_features))
            self.register_buffer('features_ema', torch.zeros(self.num_instances * num_samples, num_features))            
        else:
            raise TypeError('Cluster Memory {} is invalid!'.format(self.cm_type))

    def forward(self, inputs, targets, model_name='encoder'):
        inputs = F.normalize(inputs, dim=1).cuda()
        if self.cm_type == 'CM':
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            return loss

        elif self.cm_type == 'CMhybrid':
            outputs = cm_hybrid(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            mean, hard = torch.chunk(outputs, 2, dim=1)
            r = 0.2
            loss = 0.5 * (self.cross_entropy(hard, targets) + torch.relu(self.cross_entropy(mean, targets) - r))
            return loss
        
        elif self.cm_type == 'CMhard':
            if model_name == 'encoder':
                outputs = cm_hybrid(inputs, targets, self.features, self.momentum)
                outputs /= self.temp
                mean, hard = torch.chunk(outputs, 2, dim=1)
                r = 0.2
                loss1 = 0.5 * (self.cross_entropy(hard, targets) + torch.relu(self.cross_entropy(mean, targets) - r))
                return loss1
            elif model_name == 'encoder_ema':
                outputs = cm_hard(inputs, targets, self.features_ema, self.momentum, self.num_instances)
                outputs /= self.temp
                out_list = torch.chunk(outputs, self.num_instances, dim=1)
                out = torch.stack(out_list, dim=0)
                neg = torch.max(out, dim=0)[0]
                pos = torch.min(out, dim=0)[0]
                mask = torch.zeros_like(out_list[0]).scatter_(1, targets.unsqueeze(1), 1)
                mask_in = mask * pos + (1 - mask) * neg
                loss2 = self.cross_entropy(mask_in, targets)
                return loss2     
