import torchvision
from .u_net_resnet_50_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from src.logger import Logger
log = Logger()
import cv2

class UNetWithResnet50Encoder(nn.Module):
    
    DEPTH = 6

    def __init__(self, config=None, distribution=np.array([[0.01, 0.99], [0.01, 0.99], [0.01, 0.99]])):
        super().__init__()

        self.config = config
        if config['lossf'] == 'CEL':
            self.classes = self.config['n_classes']*2
        elif config['lossf'] == 'DICE':
            self.classes = self.config['n_classes']

        distribution[:,[0, 1]] = distribution[:,[1, 0]]
        self.distribution = torch.FloatTensor(distribution).cuda()
        self.loss_name = self.config['lossf']


        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, self.classes, kernel_size=1, stride=1)


        if self.config['optimizer'] == 'Adam':
            log.info('Using Adam optimizer')
            self.opt = optim.Adam(self.parameters(), lr=config['lr'])
        elif self.config['optimizer'] == 'SGD':
            log.info('Using SGD optimizer')
            self.opt = optim.SGD(self.parameters(), lr=config['lr'], momentum=0.9)

        if config['lossf'] == 'CEL':
            log.info('Using CEL')
            self.lossf = self.CEL_loss_final
        elif config['lossf'] == 'MSE':
            log.info('Using MSE')
            self.lossf = torch.nn.MSELoss()
        elif config['lossf'] == 'DICE':
            log.info('Using DICE')
            self.lossf = self.dice_loss_final

    def update_lr(self, epoch):

        lr = self.config['lr'] + (self.config['min_lr'] - self.config['lr'])*epoch/self.config['epoch']

        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def triangular_lr(self, iterations, epoch):

        lr_min = self.config['triangular_lr']['min_lr']
        lr_peak = self.config['triangular_lr']['peak_lr']

        lr_max = lr_peak - lr_peak*epoch/(self.config['epoch']+1)
        step_size = self.config['triangular_lr']['step_size']
        cycle = np.floor(1+iterations/(2*step_size))
        x = np.abs(iterations/step_size - 2*cycle + 1)
        lr_new= lr_min + (lr_max-lr_min)*np.maximum(0, (1-x))

        for param_group in self.opt.param_groups:
            param_group['lr'] = lr_new

        return lr_new

    def CEL_loss_final(self, x, y_, type_, what_is_annotated):

        loss = 0

        for no, type_i in enumerate(type_):

            type__ = type_i.split('-')[0]

            if type__ == 'nose':

                if 'spot' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/2

                if 'patch' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, 2:4], y_[no, :, 1], weight=self.distribution[1])#/2

            elif type__ == 'cheek':

                if 'spot' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/3

                if 'patch' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, 2:4], y_[no, :, 1], weight=self.distribution[1])#/3

                if 'wrinkle' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, 4:], y_[no, :, 2], weight=self.distribution[2])*5/2

            elif type__ == 'forehead':

                if 'spot' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/3

                if 'patch' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, 2:4], y_[no, :, 1], weight=self.distribution[1])#/3

            elif type__ == 'chin':              
                if 'spot' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/3

                if 'patch' in what_is_annotated[no][type__]:
                    loss += F.cross_entropy(x[no, :, 2:4], y_[no, :, 1], weight=self.distribution[1])#/3

        return loss/x.size()[0]

    def CEL_loss_numpy(self, x, y_, info):

        b, ch, h, w = y_.shape

        x = torch.FloatTensor(x.transpose(0, 3, 2, 1).reshape(b*w*h, ch)).cuda()
        y_ = torch.FloatTensor(y_.transpose(0, 3, 2, 1).reshape(b*h*w, ch)).cuda().long()

        _, y_ = torch.max(y_, dim=1)
        loss_c = self.lossf(x, y_).data.cpu().numpy()
        temp, indi_acc = self.accuracy(x, y_)
        temp= temp.data.cpu().numpy()
        info['Acc_indi'] = (indi_acc + info['Count']*info['Acc_indi'])/(info['Count']+1)
        info['Acc'] = (temp + info['Count']*info['Acc'])/(info['Count']+1)
        info['Loss'] = (loss_c + info['Count']*info['Loss'])/(info['Count']+1)
        info['Count'] += 1

        return loss_c

    def dice_loss_numpy(self, x, y_, info):

        b, ch, h, w = y_.shape

        x = x.transpose(0, 3, 2, 1).reshape(b*w*h, ch)
        y_ = y_.transpose(0, 3, 2, 1).reshape(b*h*w, ch)


        intersection1 = np.sum(x[:, 0]*y_[:, 0])
        intersection2 = np.sum(x[:, 1]*y_[:, 1])
        intersection3 = np.sum(x[:, 2]*y_[:, 2])
        union1 = np.sum(x[:, 0]+y_[:, 0]) - intersection1
        union2 = np.sum(x[:, 1]+y_[:, 1]) - intersection2
        union3 = np.sum(x[:, 2]+y_[:, 2]) - intersection3
        temp  = self.accuracy_numpy(x, y_)
        info['Acc'] = (temp + info['Count']*info['Acc'])/(info['Count']+1)
        loss_c = 1-(intersection1/union1 + intersection2/union2+intersection3/union3)/3
        info['Loss'] = (loss_c + info['Count']*info['Loss'])/(info['Count']+1)
        info['Count'] += 1

        return loss_c

    def dice_loss_final(self, x, y, type_):

        b, ch, h, w = y.shape

        x = x.transpose(1, 3).contiguous().view(b, w*h, ch)
        y = y.transpose(1, 3).contiguous().view(b, h*w, ch)

        loss = 0

        for no, type_i in enumerate(type_):

            type__ = type_i.split('-')[0]

            if type__ == 'nose':
                intersection = torch.sum(x[no, :, :2]*y[no, :, :2])
                union = torch.sum(x[no, :, :2]+y[no, :, :2]) - intersection
                if union.data.cpu().numpy() != 0:
	                loss += 1-(intersection/union)
            elif type__ == 'cheek':
                intersection = torch.sum(x[no, :, :]*y[no, :, :])
                union = torch.sum(x[no, :, :]+y[no, :, :]) - intersection
                if union.data.cpu().numpy() != 0:
	                loss += 1-(intersection/union)
            elif type__ == 'forehead':
                intersection = torch.sum(x[no, :, [0, 2]]*y[no, :, [0, 2]])
                union = torch.sum(x[no, :, [0, 2]]+y[no, :, [0, 2]]) - intersection
                if union.data.cpu().numpy() != 0:
	                loss += 1-(intersection/union)
            elif type__ == 'chin':
                intersection = torch.sum(x[no, :, :2]*y[no, :, :2])
                union = torch.sum(x[no, :, :2]+y[no, :, :2]) - intersection
                if union.data.cpu().numpy() != 0:
	                loss += 1-(intersection/union)

        return loss/b


    # def accuracy_numpy(self, x, y):

    #   arg = np.argmax(x, axis=0)

    #   eq = arg==y

    #   return np.mean(eq.astype(np.float32))

    def accuracy_numpy(self, x, y):

        thresholded = np.argmax(x, axis=0)
        intersection = np.sum(np.multiply(thresholded, y))
        union = np.sum(thresholded)+np.sum(y)-intersection
        if union == 0:
        	return 1
        return intersection/union

    def accuracy(self, x, y, train_=False):

        if self.loss_name == 'DICE':
            indi_acc = np.zeros(self.config['n_classes'])
            for i in range(self.config['n_classes']):
                thresholded = x[:, i].data.cpu().numpy()
                thresholded[thresholded<0.5] = 0
                thresholded[thresholded>=0.5] = 1
                indi_acc[i] = np.mean((thresholded == y.data.cpu().numpy()[:, i]).astype(np.float32))

            return np.mean(indi_acc)

        if self.loss_name == 'CEL':

            indi_acc = np.zeros(self.config['n_classes'])
            for i in range(self.config['n_classes']):
                thresholded = np.argmax(x[:, 2*i:2*i+2].data.cpu().numpy(), axis=1)
                indi_acc[i] = np.mean((thresholded == y.data.cpu().numpy()[:, i]).astype(np.float32))

            return np.mean(indi_acc)


        eq = torch.eq(arg.squeeze(), y.squeeze())
        if not train_:
            indi_acc = np.zeros(self.config['n_classes'])
            for i in range(self.config['n_classes']):
                temp = eq[np.logical_or(y==i, arg==i)]
                if temp.size()[0]!=0:
                    indi_acc[i] = np.mean(temp.data.cpu().numpy().astype(np.float32))

            return torch.mean(eq.float()), indi_acc

        return torch.mean(eq.float()).data.cpu().numpy()

    def print_info(self, info):

        log.info('The average accuracy is :', np.mean(info['Acc']))
        log.info('The current accuracy is :', info['Acc'][-1])
        log.info('The average loss is :', np.mean(info['Loss']))
        log.info('The current loss is :', info['Loss'][-1])

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools["layer_"+str(i)] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = "layer_"+str(UNetWithResnet50Encoder.DEPTH - 1 - i)
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

    def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar', best={}, indi_acc={}):
        
        if is_best:
            torch.save({'epoch': epoch_i,
                    'state_dict': self.state_dict(),
                    'optimizer': self.opt.state_dict(),
                    'seed': self.config['seed'],
                    'best': best,
                    'indi_acc': indi_acc},self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
            torch.save(info, self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
        else:
            torch.save({'epoch': epoch_i,
                    'state_dict': self.state_dict(),
                    'optimizer': self.opt.state_dict(),
                    'seed': self.config['seed'],
                    'best': best,
                    'indi_acc': indi_acc},self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
            torch.save(info, self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
            
            # shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
        
            # shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')
    def save_individual(self, info, which, epoch_i=0,filename='checkpoint.pth.tar',best={}):

        torch.save({'epoch': epoch_i,
                'state_dict': self.state_dict(),
                'optimizer': self.opt.state_dict(),
                'seed': self.config['seed'],
                'best': best},self.config['dir']['Model_Output_Best']+'/'+'indi_'+str(which)+'_'+filename)
        torch.save(info, self.config['dir']['Model_Output_Best']+'/'+'indi_'+str(which)+'_'+'info_'+filename)

            
            # shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
            # shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

    def load(self, path, path_info, Tr_te):

        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['state_dict'])

        if not self.config['optimizer_new']:
            self.opt.load_state_dict(checkpoint['optimizer'])

        if Tr_te == 'Training':
        
            return checkpoint['epoch'], torch.load(path_info), checkpoint['indi_acc']

        else:

            return checkpoint['epoch'], torch.load(path_info)

    def loss(self, pred, target, path, what_is_annotated, info):

        if self.config['lossf'] == 'DICE':

            loss_c = self.lossf(pred, target, path, what_is_annotated)

            b, ch, h, w = pred.size()

            pred = pred.transpose(1, 3).contiguous().view(b*w*h, ch)
            target = target.transpose(1, 3).contiguous().view(b*h*w, ch)

        else:

            b, ch, h, w = pred.size()

            pred = pred.transpose(1, 3).contiguous().view(b, w*h, ch)
            target = target.transpose(1, 3).contiguous().view(b, h*w, ch//2).long()

            loss_c = self.lossf(pred, target, path, what_is_annotated)

            pred = pred.view(b*w*h, ch)
            target = target.view(b*h*w, ch//2)

        if info['Keep_log']:

            info['Acc'].append(self.accuracy(pred, target, True))
            info['Loss'].append(loss_c.data.cpu().numpy())

        else:

            acc = self.accuracy(pred, target, True)
            info['Acc'] = (acc + info['Count']*info['Acc'])/(info['Count']+1)
            info['Loss'] = (loss_c.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)
            info['Acc_indi'] = (indi_acc + info['Count']*info['Acc_indi'])/(info['Count']+1)

        info['Count'] += 1

        return loss_c