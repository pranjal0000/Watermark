import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from .unet_parts import *
from src.logger import Logger
log = Logger()
import cv2

class UNet(nn.Module):

	def __init__(self, config=None, distribution=np.array([[0.01, 0.99], [0.01, 0.99], [0.01, 0.99]])):
		
		super(UNet, self).__init__()

		self.config = config
		self.inc = inconv(self.config['n_channels'], 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)
		self.up1 = up(1024, 256)
		self.up2 = up(512, 128)
		self.up3 = up(256, 64)
		self.up4 = up(128, 64)
		self.sigma = nn.Sigmoid()
		if config['lossf'] == 'CEL':
			self.classes = self.config['n_classes']*2
		elif config['lossf'] == 'DICE':
			self.classes = self.config['n_classes']
		self.outc = outconv(64, self.classes)
		# print(distribution)
		distribution[:,[0, 1]] = distribution[:,[1, 0]]
		# print(distribution)
		self.distribution = torch.FloatTensor(distribution).cuda()
		self.loss_name = self.config['lossf']

		if self.config['optimizer'] == 'Adam':
			log.info('Using Adam optimizer')
			self.opt = optim.Adam(self.parameters(), lr=config['lr'])
		elif self.config['optimizer'] == 'SGD':
			log.info('Using SGD optimizer')
			self.opt = optim.SGD(self.parameters(), lr=config['lr'], momentum=0.9)

		if config['lossf'] == 'CEL':
			log.info('Using CEL')
			# self.weighted_loss = torch.FloatTensor([[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]).cuda()
			# self.weighted_loss = self.weighted_loss/(np.sum(self.weighted_loss))
			self.lossf = self.CEL_loss_final#torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weighted_loss))
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

	def CEL_loss_final(self, x, y_, type_):

		loss = 0

		# print(x.size(), y_.size(), type_)

		for no, type_i in enumerate(type_):

			type__ = type_i.split('-')[0]

			if type__ == 'nose':

				loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/2
				loss += F.cross_entropy(x[no, :, 2:4], y_[no, :, 1], weight=self.distribution[1])#/2

			elif type__ == 'cheek':

				loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/3
				loss += F.cross_entropy(x[no, :, 2:4], y_[no, :, 1], weight=self.distribution[1])#/3
				loss += F.cross_entropy(x[no, :, 4:], y_[no, :, 2], weight=self.distribution[2])#/3

			elif type__ == 'forehead':

				loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/2
				loss += F.cross_entropy(x[no, :, 4:], y_[no, :, 2], weight=self.distribution[2])#/2

			elif type__ == 'chin':				
				loss += F.cross_entropy(x[no, :, :2], y_[no, :, 0], weight=self.distribution[0])#/2
				loss += F.cross_entropy(x[no, :, 2:4], y_[no, :, 1], weight=self.distribution[1])#/2

		# print(x.size()[0])
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
				loss += 1-(intersection/union)
			elif type__ == 'cheek':
				intersection = torch.sum(x[no, :, :]*y[no, :, :])
				union = torch.sum(x[no, :, :]+y[no, :, :]) - intersection
				loss += 1-(intersection/union)
			elif type__ == 'forehead':
				intersection = torch.sum(x[no, :, [0, 2]]*y[no, :, [0, 2]])
				union = torch.sum(x[no, :, [0, 2]]+y[no, :, [0, 2]]) - intersection
				loss += 1-(intersection/union)
			elif type__ == 'chin':
				intersection = torch.sum(x[no, :, :2]*y[no, :, :2])
				union = torch.sum(x[no, :, :2]+y[no, :, :2]) - intersection
				loss += 1-(intersection/union)

		return loss/b


	# def accuracy_numpy(self, x, y):

	# 	arg = np.argmax(x, axis=0)

	# 	eq = arg==y

	# 	return np.mean(eq.astype(np.float32))

	def accuracy_numpy(self, x, y):

		thresholded = np.argmax(x, axis=0)
		intersection = np.sum(np.multiply(thresholded, y))
		union = np.sum(thresholded)+np.sum(y)-intersection

		if union == 0:
			return 1.0

		return intersection/union

	def accuracy(self, x, y, train_=False):

		# print(x.size(), y.size(), 'HERE')
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

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)

		x = self.outc(x)

		if self.loss_name == 'DICE':
			x = self.sigma(x)
		return x

	def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar', best={}):

		
		if is_best:
			torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best},self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
		else:
			torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best},self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
			
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

	def save_individual(self, epoch_i, info, which, filename='checkpoint.pth.tar',best={}):

		torch.save({'epoch': epoch_i,
				'state_dict': self.state_dict(),
				'optimizer': self.opt.state_dict(),
				'seed': self.config['seed'],
				'best': best},self.config['dir']['Model_Output_Best']+'/'+'indi_'+str(which)+'_'+filename)
		torch.save(info, self.config['dir']['Model_Output_Best']+'/'+'indi_'+str(which)+'_'+'info_'+filename)

			
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

	def load(self, path, path_info):

		checkpoint = torch.load(path)

		self.load_state_dict(checkpoint['state_dict'])

		if not self.config['optimizer_new']:
			self.opt.load_state_dict(checkpoint['optimizer'])
		
		return checkpoint['epoch'], torch.load(path_info)

	def loss(self, pred, target, path, info):

		if self.config['lossf'] == 'DICE':

			loss_c = self.lossf(pred, target, path)

			b, ch, h, w = pred.size()

			pred = pred.transpose(1, 3).contiguous().view(b*w*h, ch)
			target = target.transpose(1, 3).contiguous().view(b*h*w, ch)

		else:

			b, ch, h, w = pred.size()

			pred = pred.transpose(1, 3).contiguous().view(b, w*h, ch)
			target = target.transpose(1, 3).contiguous().view(b, h*w, ch//2).long()

			# print("Here:",target.size())

			loss_c = self.lossf(pred, target, path)

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