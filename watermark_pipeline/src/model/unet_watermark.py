import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from unet_parts_watermark import *
#from src.logger import Logger
#log = Logger()
import cv2

class UNet(nn.Module):

	def __init__(self):
		
		super(UNet, self).__init__()

		self.inc = inconv(3, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		# self.down3 = down(256, 512)
		# self.down4 = down(512, 512)
		# self.up1 = up(1024, 256)
		# self.up2 = up(512+256, 128)
		self.up3 = up(256+128, 64)
		self.up4 = up(128, 64)
		self.sigma = nn.Tanh()
		self.outc = outconv(64, 3)

		self.lr=0.01	
		# self.criterion = 
		self.opt = optim.Adam(self.parameters(), self.lr)


	def forward(self, inp):
		x1 = self.inc(inp)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		# x4 = self.down3(x3)
		# x5 = self.down4(x4)
		# x = self.up1(x5, x4)
		# x = self.up2(x4, x3)
		x = self.up3(x3, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		x = self.sigma(x) + inp

		return x
















































	#def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar', best={}):

		
	#	if is_best:
	#		torch.save({'epoch': epoch_i,
	#				'state_dict': self.state_dict(),
	#				'optimizer': self.opt.state_dict(),
	#				'seed': self.config['seed'],
	#				'best': best},self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
	#		torch.save(info, self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
	#	else:
	#		torch.save({'epoch': epoch_i,
	#				'state_dict': self.state_dict(),
	#				'optimizer': self.opt.state_dict(),
	#				'seed': self.config['seed'],
	#				'best': best},self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
	#		torch.save(info, self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
			
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

#	def save_individual(self, epoch_i, info, which, filename='checkpoint.pth.tar',best={}):
#
#		torch.save({'epoch': epoch_i,
#				'state_dict': self.state_dict(),
#				'optimizer': self.opt.state_dict(),
#				'seed': self.config['seed'],
#				'best': best},self.config['dir']['Model_Output_Best']+'/'+'indi_'+str(which)+'_'+filename)
#		torch.save(info, self.config['dir']['Model_Output_Best']+'/'+'indi_'+str(which)+'_'+'info_'+filename)

			
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
			# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

#	def load(self, path, path_info):

#		checkpoint = torch.load(path)

#		self.load_state_dict(checkpoint['state_dict'])

#		if not self.config['optimizer_new']:
#			self.opt.load_state_dict(checkpoint['optimizer'])
#		
#		return checkpoint['epoch'], torch.load(path_info)

	#def loss(self, pred, target, path, info):
	

#			b, ch, h, w = pred.size()
#
#			pred = pred.transpose(1, 3).contiguous().view(b, w*h, ch)
#			target = target.transpose(1, 3).contiguous().view(b, h*w, ch//2).long()

			# print("Here:",target.size())

#			loss_c = self.lossf(pred, target, path)

#			pred = pred.view(b*w*h, ch)
#			target = target.view(b*h*w, ch//2)

#		if info['Keep_log']:

#			info['Acc'].append(self.accuracy(pred, target, True))
#			info['Loss'].append(loss_c.data.cpu().numpy())

#		else:

	#		acc = self.accuracy(pred, target, True)
	#		info['Acc'] = (acc + info['Count']*info['Acc'])/(info['Count']+1)
	#		info['Loss'] = (loss_c.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)
	#		info['Acc_indi'] = (indi_acc + info['Count']*info['Acc_indi'])/(info['Count']+1)
#
#		info['Count'] += 1

#		return loss_c