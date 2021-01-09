import os
import sys
import numpy as np
import torch
from scipy import io as sio
from torch import nn as nn
from torch.optim import lr_scheduler


def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    #print('num pos', num_positive)
    #print('num neg', num_negative)
    #print(1.0 * num_negative / (num_positive + num_negative), 1.1 * num_positive / (num_positive + num_negative))

    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_negative + num_positive)


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()

    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


def get_model_params(net):

    #tune lr
    net_parameters_id = {}

    for pname, p in net.rcf.named_parameters():
        if pname in ['conv1_1.weight','conv1_2.weight',
                    'conv2_1.weight','conv2_2.weight',
                    'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                    'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
            print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias','conv1_2.bias',
                    'conv2_1.bias','conv2_2.bias',
                    'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                    'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
            print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
            print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
            print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                    'conv2_1_down.weight','conv2_2_down.weight',
                    'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                    'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                    'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
            print(pname, 'lr:0.1 de:1')
            if 'conv_down_1-5.weight' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.weight'] = []
            net_parameters_id['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                    'conv2_1_down.bias','conv2_2_down.bias',
                    'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                    'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                    'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
            print(pname, 'lr:0.2 de:0')
            if 'conv_down_1-5.bias' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.bias'] = []
            net_parameters_id['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                    'score_dsn4.weight','score_dsn5.weight']:
            print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                    'score_dsn4.bias','score_dsn5.bias']:
            print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            print(pname, 'lr:0.001 de:1')
            if 'score_final.weight' not in net_parameters_id:
                net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            print(pname, 'lr:0.002 de:0')
            if 'score_final.bias' not in net_parameters_id:
                net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)

    for pname, p in net.crf.named_parameters():

        if pname in ['cost_valid', 'cost_invalid']:
            print(pname, p)
            if 'crf_costs' not in net_parameters_id:
                net_parameters_id['crf_costs'] = []
            net_parameters_id['crf_costs'].append(p)

    return net_parameters_id


def get_opt(lr, momentum, weight_decay, net_parameters_id):

    optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight']      , 'lr': lr*1    , 'weight_decay': weight_decay},
            {'params': net_parameters_id['conv1-4.bias']        , 'lr': lr*2    , 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight']        , 'lr': lr*100  , 'weight_decay': weight_decay},
            {'params': net_parameters_id['conv5.bias']          , 'lr': lr*200  , 'weight_decay': 0.},
            {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': lr*0.1  , 'weight_decay': weight_decay},
            {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': lr*0.2  , 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr*0.01 , 'weight_decay': weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': lr*0.02 , 'weight_decay': 0.},
            {'params': net_parameters_id['score_final.weight']  , 'lr': lr*0.001, 'weight_decay': weight_decay},
            {'params': net_parameters_id['score_final.bias']    , 'lr': lr*0.002, 'weight_decay': 0.},
            {'params': net_parameters_id['crf_costs']           , 'lr': lr*1000 , 'weight_decay': weight_decay},
            #{'params': net_parameters_id['crf_costs']           , 'lr': lr*10000 , 'weight_decay': weight_decay},#increase because of smaller start LR
        ], lr=lr, momentum=momentum, weight_decay=weight_decay)


    return optimizer


def get_scheduler(optimizer, stepsize, gamma):

    scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

    return scheduler

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()


def make_optim(model, lr, momentum, decay):
    optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    return optim