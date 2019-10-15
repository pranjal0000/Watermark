import sys
from torchvision import transforms
import torch.utils.data as data
import torch
from torch.autograd import Variable
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import shutil
from scipy.misc import imresize
import time
import json

from .read_yaml import read_yaml
from .model.resnet import resnet152
from .model.alexnet import alexnet
from .logger import Logger
from .model import generic_model
from .model.unet import UNet
from .model.u_net_resnet_50_encoder import UNetWithResnet50Encoder 
from .prepare_metadata.metadata import metadata
from .generate_json import generate_json

log = Logger()

class dl_model():

	def __init__(self, model, Training_Testing = 'Training', target_transform=None, train_transform=None, test_transform=None):
		
		self.config = self.get_config()
		self.Training_Testing = Training_Testing

		self.seed()

		self.gen_json = generate_json()

		if Training_Testing != 'ensemble':

			self.cuda = self.config['cuda'] and torch.cuda.is_available()

			self.plot_training = {'Loss' : [], 'Acc' : []}	
			self.plot_testing = {'Loss' : [], 'Acc' : []}

			self.get_transforms(target_transform, train_transform, test_transform)

			if self.config['dataloader'] == 'Pytorch':

				from .data_loader import DataLoader
			
				self.train_data = DataLoader(self.config['train'], transform=self.train_transform, target_transform = self.target_transform)
				self.train_data_loader = data.DataLoader(self.train_data, batch_size=self.config['train']['batch_size'], shuffle=True, num_workers=self.config['train']['cpu_alloc'])
				
				self.test_data = DataLoader(self.config['test'], transform=self.test_transform, target_transform = self.target_transform)
				self.test_data_loader = data.DataLoader(self.test_data, batch_size=self.config['test']['batch_size'], shuffle=False, num_workers=self.config['test']['cpu_alloc'])


			else:

				from .data_loader import own_DataLoader

				self.train_data_loader = own_DataLoader(self.config['train'], transform=self.train_transform, target_transform = self.target_transform)


				if Training_Testing != 'test_one':
					self.test_data_loader = own_DataLoader(self.config['test'], transform=self.test_transform, target_transform = self.target_transform)
				else:
					# print("Here")
					self.test_data_loader = own_DataLoader(self.config['test_one'], transform=self.test_transform, target_transform = self.target_transform)

			self.model = self.get_model(model, self.train_data_loader.distribution)

			self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}
			self.testing_info = {'Acc': 0, 'count': 0}
			
			self.model_best = {'Loss': sys.float_info.max, 'Acc': 0.0, 'Acc_indi': np.zeros(self.config['n_classes']+1)}

			if self.cuda:
				self.model.cuda()

			self.epoch_start = 0
			self.start_no = 0
			self.lrs=[]

			self.region_to_type = {'cheek':['spot','patch','wrinkle'], 'nose':['spot','patch'], 'forehead':['spot'], 'chin':['spot','patch']}
			self.type_to_region = {'spot':['cheek','nose','forehead','chin'], 'patch':['cheek','nose','chin'], 'wrinkle':['cheek']}

			self.lrs = []
			self.iterations = []
			self.iterations_count = 0

			# self.indi_acc = {'spot':0.25707033955711156,'patch':0.6679174484052534, 'wrinkle':0.5054860830139218}
			self.indi_acc = {'spot':0, 'patch':0, 'wrinkle':0}

			if self.config['PreTrained_model']['check'] == True or Training_Testing=='testing':

				self.model_best = torch.load(self.config['PreTrained_model']['checkpoint_best'])['best']
				if Training_Testing == 'Training':
					self.epoch_start, self.training_info, self.indi_acc = self.model.load(self.config['PreTrained_model']['checkpoint'], self.config['PreTrained_model']['checkpoint_info'], Training_Testing)
				else:
					self.epoch_start, self.training_info = self.model.load(self.config['PreTrained_model']['checkpoint'], self.config['PreTrained_model']['checkpoint_info'], Training_Testing)

				self.start_no = int(self.config['PreTrained_model']['checkpoint'].split('/')[-1].split('_')[0])
				self.epoch_start = int(self.config['PreTrained_model']['checkpoint'].split('/')[-1].split('_')[1])
				log.info('Till here Individual Accuracy = ', self.indi_acc)

				if Training_Testing == 'Training':

					self.plot_training['Loss'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_training_loss.npy'))
					self.plot_training['Acc'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_training_accuracy.npy'))
					self.plot_testing['Acc'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_testing_accuracy.npy'))
				
					self.model.update_lr(self.epoch_start)
				log.info('Loaded the model')

		else:

			log.info('Ensembling')
	
	def get_model(self, model, distribution):

		if model == 'UNet':
			log.info("UNET")
			channels, classes = self.config['n_channels'], len(os.listdir(self.config['train']['dir']+'/Annotations')) # +1 for Background
			return UNet(config=self.config, distribution=distribution)
		elif model == 'ResNet_UNet':
			log.info("RESNET_UNET")
			channels, classes = self.config['n_channels'], len(os.listdir(self.config['train']['dir']+'/Annotations')) # +1 for Background
			return UNetWithResnet50Encoder(config=self.config, distribution=distribution)
		elif model == 'ResNet':
			return resnet152(pretrained=self.config['PreTrained_net'], config=self.config)
		
		elif model == 'AlexNet':
			return alexnet(pretrained=self.config['PreTrained_net'], config=self.config)
		
		else:
			log.info("Can't find model")

	def get_transforms(self, target_transform=None, train_transform=None, test_transform=None):

		if self.config['train']['transform'] == False or train_transform == None:
			self.train_transform = transforms.Compose([
											transforms.ColorJitter(brightness=self.config['augmentation']['brightness'], contrast=self.config['augmentation']['contrast'], saturation=self.config['augmentation']['saturation'], hue=self.config['augmentation']['hue']),
											# transforms.ToTensor(),
											])
		else:
			self.train_transform = train_transform

		if self.config['test']['transform'] == False or test_transform == None:
			self.test_transform = transforms.Compose([
											 # transforms.ToTensor(),
											 ])
		else:
			self.test_transform = test_transform
		
		if self.config['target_transform'] == False or target_transform == None:

			self.target_transform = transforms.Compose([
											 transforms.ToTensor(),
											 ])
		else:
			self.target_transform = target_transform

	def get_config(self):

		return read_yaml()

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])

	def __str__(self):

		return str(self.config)

	def convert_argmax_to_channels(self, temp, masks):

		t_show = np.zeros([temp.shape[0], temp.shape[1], masks]).astype(np.uint8)

		for __i in range(t_show.shape[0]):
			for __j in range(t_show.shape[1]):
				t_show[__i, __j, temp[__i, __j]] = 255

		return t_show

	def start_training(self):

		self.model.requires_grad = True

		self.model.train()

		self.model.opt.zero_grad()

	def start_testing(self):

		self.model.requires_grad = False

		self.model.eval()

	def show_graph(self, epoch, no):

		plt.clf()

		plt.subplot(211)
		plt.plot(self.plot_training['Loss'], color='red')
		plt.title('Upper Plot: Loss, Red:Training, Blue:Testing\nLower Plot: Accuracy, Red:Training, Blue:Testing')
		plt.subplot(212)
		plt.plot(self.plot_training['Acc'], color='red')
		plt.plot(self.plot_testing['Acc'], color='blue')
		#plt.pause(0.1)
		plt.savefig(self.config['dir']['Plots']+'/'+str(epoch)+'_'+str(no)+'.png')

	def test_module(self, epoch_i, no):

		self.test_model()

		self.plot_training['Loss'].append(np.mean(self.training_info['Loss']))
		self.plot_training['Acc'].append(np.mean(self.training_info['Acc']))

		self.show_graph(epoch_i, no)
		self.start_training()

	def train_model(self):

		try:

			# self.test_module(100, 100)

			self.start_training()			

			for epoch_i in range(self.epoch_start, self.config['epoch']+1):

				log.info('Starting epoch : ', epoch_i)

				for no, (data, target, path, what_is_annotated) in enumerate(self.train_data_loader):

					# print(target.shape)

					self.iterations_count += 1

					# plt.subplot(2,1,1)
					# plt.imshow(data[0].data.cpu().numpy().transpose(1, 2, 0))
					# plt.subplot(2,1,2)
					# plt.imshow(target[0].data.cpu().numpy().transpose(1, 2, 0))
					# plt.pause(1)
					# print(data[0][:, 200:600, 200:600])
					# plt.imshow(data[0].cpu().numpy().transpose(1, 2, 0))
					# plt.show()
					# plt.imshow(target[0].cpu().numpy().transpose(1, 2, 0))
					# plt.show()
			
					target = target[:, :, self.config['padding']:self.config['padding']+512, self.config['padding']:self.config['padding']+512]

					# new_lr = self.model.triangular_lr(self.iterations_count, epoch_i)
					# print(new_lr)
					# self.iterations.append(self.iterations_count)
					# self.lrs.append(new_lr)

					# plt.scatter(self.iterations_count,new_lr)
					# plt.pause(1)

					data, target = Variable(data), Variable(target)

					if self.cuda:

						data, target = data.cuda(), target.cuda()

					data = self.model(data)[:, :, self.config['padding']:self.config['padding']+512, self.config['padding']:self.config['padding']+512]

					loss = self.model.loss(data, target, path, what_is_annotated, self.training_info)

					loss.backward()

					if (self.start_no + no)%self.config['update_config']==0 and (self.start_no + no)!=0:

						prev_config = self.config

						self.config = self.get_config()
						self.train_data_loader.config = self.config
						self.model.config = self.config

						if self.config['lr']!=prev_config['lr']:
							log.info('Learning Rate Changed from ', prev_config['lr'], ' to ', self.config['lr'])

					if (self.start_no + no)%self.config['cummulative_batch_steps']==0 and (self.start_no + no)!=0:

						self.model.opt.step()

						self.model.opt.zero_grad()

					# # 	plt.clf()
					# # 	plt.imshow(np.concatenate((np.argmax(data[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0:2], axis=2), target[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0]), axis=0))
					# # 	plt.pause(1)
					# # 	plt.imshow(np.concatenate((np.argmax(data[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 2:4], axis=2), target[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 1]), axis=0))
					# # 	plt.pause(1)
					# # 	plt.imshow(np.concatenate((np.argmax(data[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 4:6], axis=2), target[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 2]), axis=0))
					# # 	plt.pause(1)

					if (self.start_no + no) == len(self.train_data_loader) - 1:
						break

				self.model.print_info(self.training_info)
				log.info()
				self.start_no = 0

				self.test_module(epoch_i, 0)

				if epoch_i%10 == 0 and epoch_i!=0:

					self.model.save(no=(self.start_no + no), epoch_i=epoch_i, info = self.training_info, best=self.model_best, indi_acc=self.indi_acc)
							
					np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_testing_accuracy.npy', self.plot_testing['Acc'])
					np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_training_loss.npy', self.plot_training['Loss'])
					np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_training_accuracy.npy', self.plot_training['Acc'])

				self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}

			return True

		except KeyboardInterrupt:

			return False

	def reverse_aspect_resize(self, image, original_h, original_w):

		# exit(0)

		if len(image.shape) == 3:
			original = np.copy(image)
			
			output = []
			for i in range(original.shape[2]):
				image = original[:, :, i]
				mini = np.min(image)
				maxi = np.max(image)
				image = ((image - mini)/(maxi - mini)*255).astype(np.uint8)
				if original_h > original_w:

					original_shaped = imresize(image, (original_h, original_h))
					original_shaped = original_shaped[:, (original_h - original_w)//2 : (original_h + original_w)//2]

				else:

					original_shaped = imresize(image, (original_w, original_w))
					original_shaped = original_shaped[(original_w - original_h)//2 : (original_w + original_h)//2, :]

				original_shaped = original_shaped.astype(np.float32)/255*(maxi - mini) + mini
				output.append(original_shaped)
			# print('Here')
			# plt.imshow(np.argmax(image, axis=2))
			# plt.imshow(np.argmax(np.array(output).transpose(1, 2, 0), axis=2))
			# plt.show()
			return np.array(output).transpose(1, 2, 0)

		else:

			if original_h > original_w:

				original_shaped = imresize(image, (original_h, original_h))
				original_shaped = original_shaped[:, (original_h - original_w)//2 : (original_h + original_w)//2]

			else:

				original_shaped = imresize(image, (original_w, original_w))
				original_shaped = original_shaped[(original_w - original_h)//2 : (original_w + original_h)//2, :]

		return original_shaped

	def test_model(self):

		log.info('Testing Mode')

		try:

			self.start_testing()	

			indi_acc = {'spot': [], 'patch': [], 'wrinkle': []}	

			for no, (data, mask, target, path, what_is_annotated) in enumerate(self.test_data_loader):

				# data has structure - {'cheek': torch.FloatTensor([3, 512, 512]),  ..., 'co_ordinates':{'cheek': [x, y, w, h], ..., 'original' : [height, width, 3]}}
				# target has structure - {'cheek': torch.FloatTensor([3, 512, 512]),  ...}
				# mask has structure - {'cheek': np.array([512, 512]),  ...} range(0, 1)

				output = {
							'cheek-1':    np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]),
							'cheek-2':    np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]), 
							'nose':     np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]),
							'forehead': np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]),
							'chin':     np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']])
						 } 

				output_accuracy = {
										'cheek-1':    [0, 0, 0], 
										'cheek-2':    [0, 0, 0], 
										'nose':     [0, 0, 0],
										'forehead': [0, 0, 0],
										'chin':     [0, 0, 0]
								  }

				final_output = np.zeros(data['co_ordinates']['original'])
				continuous_output = np.zeros([data['co_ordinates']['original'][0], data['co_ordinates']['original'][1], 6])

				for key in target.keys():

					data_key, target_key = Variable(data[key]), target[key]

					if self.cuda:
						data_key = data_key.cuda()

					output[key] = self.model(data_key).data.cpu().numpy()[0, :, self.config['padding']:512+self.config['padding'], self.config['padding']:self.config['padding']+512]

					del data_key

					if key == 'cheek-1':

						cheek_co = data['co_ordinates']['cheek-1']

						if 'spot' in what_is_annotated['cheek']:

							output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
							output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0] = output_0

							cont_output_0 = output[key][0:2]*mask[key][None, :, :]
							cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0:2] = cont_output_0

						if 'patch' in what_is_annotated['cheek']:

							output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
							output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 1] = output_1

							cont_output_1 = output[key][2:4]*mask[key][None, :, :]
							cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2:4] = cont_output_1

							plt.imshow(cont_output_1[:, :, 1])
							plt.show()

						if 'wrinkle' in what_is_annotated['cheek']:

							output_2 = np.argmax(output[key][4:6, :, :], axis=0)*mask[key]
							output_2 = self.reverse_aspect_resize(output_2, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2] = output_2

							cont_output_2 = output[key][4:6]*mask[key][None, :, :]
							cont_output_2 = self.reverse_aspect_resize(cont_output_2.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 4:6] = cont_output_2

					elif key == 'cheek-2':

						cheek_co = data['co_ordinates']['cheek-2']

						if 'spot' in what_is_annotated['cheek']:

							output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
							output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0] = output_0

							cont_output_0 = output[key][0:2]*mask[key][None, :, :]
							cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0:2] = cont_output_0

						if 'patch' in what_is_annotated['cheek']:

							output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
							output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 1] = output_1

							cont_output_1 = output[key][2:4]*mask[key][None, :, :]
							cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2:4] = cont_output_1

						if 'wrinkle' in what_is_annotated['cheek']:
							
							output_2 = np.argmax(output[key][4:6, :, :], axis=0)*mask[key]
							output_2 = self.reverse_aspect_resize(output_2, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2] = output_2

							cont_output_2 = output[key][4:6]*mask[key][None, :, :]
							cont_output_2 = self.reverse_aspect_resize(cont_output_2.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 4:6] = cont_output_2

					elif key == 'nose':

						nose_co = data['co_ordinates'][key]

						if 'spot' in what_is_annotated['nose']:

							output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
							output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 0] = output_0

							cont_output_0 = output[key][0:2]*mask[key][None, :, :]
							cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 0:2] = cont_output_0

						if 'patch' in what_is_annotated['nose']:

							output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
							output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 1] = output_1

							cont_output_1 = output[key][2:4]*mask[key][None, :, :]
							cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 2:4] = cont_output_1

					elif key == 'forehead':

						forehead_co = data['co_ordinates'][key]

						if 'spot' in what_is_annotated['forehead']:

							output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
							output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 0] = output_0

							cont_output_0 = output[key][0:2]*mask[key][None, :, :]
							cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 0:2] = cont_output_0

						if 'patch' in what_is_annotated['forehead']:

							output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
							output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 1] = output_1

							cont_output_1 = output[key][2:4]*mask[key][None, :, :]
							cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 2:4] = cont_output_1

					elif key == 'chin':				

						chin_co = data['co_ordinates'][key]

						if 'spot' in what_is_annotated['chin']:

							output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
							output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 0] = output_0

							cont_output_0 = output[key][0:2]*mask[key][None, :, :]
							cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 0:2] = cont_output_0

						if 'patch' in what_is_annotated['chin']:

							output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
							output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							final_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 1] = output_1

							cont_output_1 = output[key][2:4]*mask[key][None, :, :]
							cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
							continuous_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 2:4] = cont_output_1

					output_accuracy[key][0] = self.model.accuracy_numpy(output[key][0:2, :, :], target_key[:, :, 0])
					output_accuracy[key][1] = self.model.accuracy_numpy(output[key][2:4, :, :], target_key[:, :, 1])
					output_accuracy[key][2] = self.model.accuracy_numpy(output[key][4:6, :, :], target_key[:, :, 2])

				spot_acc, patch_acc, wrinkle_acc = 0,0,0
				spot_count, patch_count, wrinkle_count = 0,0,0

				for region, acc in output_accuracy.items():
					if 'spot' in what_is_annotated[region.split('-')[0]]:
						spot_acc += acc[0]
						spot_count += 1
					if region in ['cheek-1','cheek-2','nose','chin']:
						if 'patch' in what_is_annotated[region.split('-')[0]]:
							patch_acc += acc[1]
							patch_count += 1
						
					if region in ['cheek-1','cheek-2']:
						if 'wrinkle' in what_is_annotated[region.split('-')[0]]:
							wrinkle_acc += acc[2]
							wrinkle_count += 1

				spot_acc /= spot_count
				patch_acc /= patch_count
				wrinkle_acc /= wrinkle_count

				indi_acc['spot'].append(spot_acc)
				indi_acc['patch'].append(patch_acc)
				indi_acc['wrinkle'].append(wrinkle_acc)

				total_accuracy = 0

				total_accuracy += np.sum(output_accuracy['cheek-1'])
				total_accuracy += np.sum(output_accuracy['cheek-2'])
				total_accuracy += np.sum(output_accuracy['nose'][0:2])
				total_accuracy += output_accuracy['forehead'][0] + output_accuracy['forehead'][2]
				total_accuracy += np.sum(output_accuracy['chin'][0:2])
				total_accuracy = total_accuracy/12

				log.info()
				log.info('On Image: ', path)
				log.info('cheek-1', output_accuracy['cheek-1'])
				log.info('cheek-1', output_accuracy['cheek-2'])
				log.info('nose', output_accuracy['nose'])
				log.info('forehead', output_accuracy['forehead'])
				log.info('chin', output_accuracy['chin'])


				if not os.path.exists(self.config['dir']['Output']+'/'+path):
					os.mkdir(self.config['dir']['Output']+'/'+path)

				base_path = self.config['dir']['Output']+'/'+path

				plt.imsave(base_path+'/'+path+'_argmax.png', final_output.astype(np.uint8))
				np.save(base_path+'/'+path+'_continuous.npy', continuous_output)
				with open(base_path+'/'+path+'_data.json', 'w') as fp:
					json.dump(self.gen_json.generate(final_output.astype(np.uint8), path), fp, indent=4, separators=(',', ': '), sort_keys=True)
				
				self.testing_info['Acc'] += total_accuracy
				self.testing_info['count'] += 1

			self.testing_info['Acc'] = self.testing_info['Acc']/self.testing_info['count']

			log.info()

			if self.Training_Testing =='Training':
			
				if np.mean(indi_acc['spot']) > self.indi_acc['spot']:
					self.model.save_individual(info=self.training_info, which=0)
					self.indi_acc['spot'] = np.mean(indi_acc['spot'])
					log.info("New best Dice score for spots:",self.indi_acc['spot'])

				if np.mean(indi_acc['wrinkle']) > self.indi_acc['wrinkle']:
					self.model.save_individual(info=self.training_info, which=2)
					self.indi_acc['wrinkle'] = np.mean(indi_acc['wrinkle'])
					log.info("New best Dice score for wrinkle:",self.indi_acc['wrinkle'])

				if np.mean(indi_acc['patch']) > self.indi_acc['patch']:
					self.model.save_individual(info=self.training_info, which=1)
					self.indi_acc['patch'] = np.mean(indi_acc['patch'])
					log.info("New best Dice score for patches:",self.indi_acc['patch'])

			log.info('Current Spots Dice Score: ', np.mean(indi_acc['spot']))
			log.info('Current Patch Dice Score: ', np.mean(indi_acc['patch']))
			log.info('Current Wrinkle Dice Score: ', np.mean(indi_acc['wrinkle']))
			log.info()

			log.info('Test Results\n\n', )

			if self.Training_Testing =='Training':

				if self.testing_info['Acc'] > self.model_best['Acc']:

					log.info("New best model found")
					self.model_best['Acc'] = self.testing_info['Acc']
					
					self.model.save(no=0, epoch_i=0, info = self.testing_info, best=self.model_best, is_best=True)

					if os.path.exists(self.config['dir']['Output_Best']):
						shutil.rmtree(self.config['dir']['Output_Best'])
					shutil.copytree(self.config['dir']['Output'], self.config['dir']['Output_Best'])

				self.plot_testing['Acc'].append(self.testing_info['Acc'])

			log.info('\nTesting Completed successfully: Average accuracy = ', self.testing_info['Acc'])

			self.testing_info = {'Acc': 0, 'count': 0}

			

			return True

		except KeyboardInterrupt:

			log.info('Testing Interrupted')

			return False

	def ensemble(self):

		def softmax_argmax(image):
			t = np.exp(image)
			return np.argmax(t/np.sum(t, axis=2)[:, :, None], axis=2)*(np.abs(image[:, :, 0]) >= 0.05).astype(np.float32)

		def combine(single_image):

			spots = np.zeros([single_image[0].shape[0], single_image[0].shape[1]])
			patch = np.zeros([single_image[0].shape[0], single_image[0].shape[1]])
			wrinkle = np.zeros([single_image[0].shape[0], single_image[0].shape[1]])

			for i in single_image:

				spots += softmax_argmax(i[:, :, 0:2])
				patch += softmax_argmax(i[:, :, 2:4])
				wrinkle += softmax_argmax(i[:, :, 4:6])

			if self.config['ensemble_way'] == 'avg_voting':

				spots /= len(single_image)
				patch /= len(single_image)
				wrinkle /= len(single_image)

			elif self.config['ensemble_way'] == 'max_voting':

				spots[spots>len(single_image)//2] = 1
				patch[patch>len(single_image)//2] = 1
				wrinkle[wrinkle>len(single_image)//2] = 1

			return (np.concatenate((spots[:, :, None], patch[:, :, None], wrinkle[:, :, None]), axis=2)*255).astype(np.uint8)

		continuous_output = {}

		for no, path in self.config['ensemble'].items():

			image_names = os.listdir(path)
			for i in image_names:
				if i in continuous_output.keys():
					continuous_output[i].append(np.load(path+'/'+i+'/'+i+'_continuous.npy'))
				else:
					continuous_output[i] = [np.load(path+'/'+i+'/'+i+'_continuous.npy')]

		for i in continuous_output.keys():

			log.info('Saving: ', i)

			to_save = combine(continuous_output[i])

			if not os.path.exists(self.config['dir']['Ensemble_Output']+'/'+i):
				os.mkdir(self.config['dir']['Ensemble_Output']+'/'+i)

			plt.imsave(self.config['dir']['Ensemble_Output']+'/'+i+'/'+i+'.png', to_save)

			with open(self.config['dir']['Ensemble_Output']+'/'+i+'/'+i+'_data.json', 'w') as fp:
				json.dump(self.gen_json.generate(to_save, i), fp, indent=4, separators=(',', ': '), sort_keys=True)

	def test_one_image(self):

		meta_obj = metadata(config_file=self.config, mode='test_one')
		# meta_obj.generate_face_crops(mode='test_one')
		# meta_obj.generate_crops()
		# meta_obj.load_save()

		log.info('Testing_One Mode')

		try:

			self.start_testing()	

			for no, (data, mask, target, path) in enumerate(self.test_data_loader):

				# data has structure - {'cheek': torch.FloatTensor([3, 512, 512]),  ..., 'co_ordinates':{'cheek': [x, y, w, h], ..., 'original' : [height, width, 3]}}
				# target has structure - {'cheek': torch.FloatTensor([3, 512, 512]),  ...}
				# mask has structure - {'cheek': np.array([512, 512]),  ...} range(0, 1)

				output = {
							'cheek-1':    np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]),
							'cheek-2':    np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]), 
							'nose':     np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]),
							'forehead': np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']]),
							'chin':     np.zeros([6, 2*(self.config['padding'])+self.config['image_size'], 2*(self.config['padding'])+self.config['image_size']])
						 } 

				final_output = np.zeros(data['co_ordinates']['original'])
				continuous_output = np.zeros([data['co_ordinates']['original'][0], data['co_ordinates']['original'][1], 6])

				for key in target.keys():

					data_key, target_key = Variable(data[key]), target[key]

					if self.cuda:
						data_key = data_key.cuda()

					output[key] = self.model(data_key).data.cpu().numpy()[0, :, self.config['padding']:512+self.config['padding'], self.config['padding']:self.config['padding']+512]

					del data_key

					if key == 'cheek-1':

						cheek_co = data['co_ordinates']['cheek-1']

						output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
						output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0] = output_0

						cont_output_0 = output[key][0:2]*mask[key][None, :, :]
						cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0:2] = cont_output_0

						output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
						output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 1] = output_1

						cont_output_1 = output[key][2:4]*mask[key][None, :, :]
						cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2:4] = cont_output_1

						# plt.imshow(cont_output_1[:, :, 1])
						# plt.show()

						output_2 = np.argmax(output[key][4:6, :, :], axis=0)*mask[key]
						output_2 = self.reverse_aspect_resize(output_2, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2] = output_2

						cont_output_2 = output[key][4:6]*mask[key][None, :, :]
						cont_output_2 = self.reverse_aspect_resize(cont_output_2.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 4:6] = cont_output_2

					elif key == 'cheek-2':

						cheek_co = data['co_ordinates']['cheek-2']

						output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
						output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0] = output_0

						cont_output_0 = output[key][0:2]*mask[key][None, :, :]
						cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 0:2] = cont_output_0


						output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
						output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 1] = output_1

						cont_output_1 = output[key][2:4]*mask[key][None, :, :]
						cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2:4] = cont_output_1
						
						output_2 = np.argmax(output[key][4:6, :, :], axis=0)*mask[key]
						output_2 = self.reverse_aspect_resize(output_2, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 2] = output_2

						cont_output_2 = output[key][4:6]*mask[key][None, :, :]
						cont_output_2 = self.reverse_aspect_resize(cont_output_2.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[cheek_co[1]:cheek_co[1] + cheek_co[3], cheek_co[0]:cheek_co[0] + cheek_co[2], 4:6] = cont_output_2

					elif key == 'nose':

						nose_co = data['co_ordinates'][key]

						output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
						output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 0] = output_0

						cont_output_0 = output[key][0:2]*mask[key][None, :, :]
						cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 0:2] = cont_output_0

						output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
						output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 1] = output_1

						cont_output_1 = output[key][2:4]*mask[key][None, :, :]
						cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[nose_co[1]:nose_co[1] + nose_co[3], nose_co[0]:nose_co[0] + nose_co[2], 2:4] = cont_output_1

					elif key == 'forehead':

						forehead_co = data['co_ordinates'][key]

						output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
						output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 0] = output_0

						cont_output_0 = output[key][0:2]*mask[key][None, :, :]
						cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 0:2] = cont_output_0


						output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
						output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 1] = output_1

						cont_output_1 = output[key][2:4]*mask[key][None, :, :]
						cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[forehead_co[1]:forehead_co[1] + forehead_co[3], forehead_co[0]:forehead_co[0] + forehead_co[2], 2:4] = cont_output_1

					elif key == 'chin':				

						chin_co = data['co_ordinates'][key]

						output_0 = np.argmax(output[key][0:2, :, :], axis=0)*mask[key]
						output_0 = self.reverse_aspect_resize(output_0, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 0] = output_0

						cont_output_0 = output[key][0:2]*mask[key][None, :, :]
						cont_output_0 = self.reverse_aspect_resize(cont_output_0.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 0:2] = cont_output_0

						output_1 = np.argmax(output[key][2:4, :, :], axis=0)*mask[key]
						output_1 = self.reverse_aspect_resize(output_1, data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						final_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 1] = output_1

						cont_output_1 = output[key][2:4]*mask[key][None, :, :]
						cont_output_1 = self.reverse_aspect_resize(cont_output_1.transpose(1, 2, 0), data['co_ordinates'][key][3], data['co_ordinates'][key][2])
						continuous_output[chin_co[1]:chin_co[1] + chin_co[3], chin_co[0]:chin_co[0] + chin_co[2], 2:4] = cont_output_1

				if not os.path.exists(self.config['test_one']['dir']+'/Results'):
					os.mkdir(self.config['test_one']['dir']+'/Results')

				base_path = self.config['test_one']['dir']+'/Results'

				plt.imsave(base_path+'/'+path+'_argmax.png', final_output.astype(np.uint8))
				np.save(base_path+'/'+path+'_continuous.npy', continuous_output)
				with open(base_path+'/'+path+'_data.json', 'w') as fp:
					json.dump(self.gen_json.generate(final_output.astype(np.uint8), path+'.png'), fp, indent=4, separators=(',', ': '), sort_keys=True)		

			return True

		except KeyboardInterrupt:

			log.info('Testing Interrupted')

			return False


