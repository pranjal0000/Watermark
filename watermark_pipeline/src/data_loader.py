import torch.utils.data as data
import torch
import random
import pickle

import os
import numpy as np

from scipy.misc import imresize
from PIL import Image

import cv2
import matplotlib.pyplot as plt
from .logger import Logger
from .face_detection import face_detect
import json

log = Logger()

class own_DataLoader():

	def __init__(self, config, **kwargs):

		self.config = config
		self.img_size = config['image_size']
		self.root = config['dir']
		self.Type = config['Type']
		self.image_size = config['image_size']
		self.IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif']
		self.transform = kwargs['transform']
		self.target_transform = kwargs['target_transform']
		self.batchsize = config['batch_size']

		# self.choices = {'spot':['cheek','forehead'], 'patch':['cheek','forehead'],'wrinkle':['cheek','forehead']}

		self.names = ['spot', 'patch', 'wrinkle']

		if not config['loader']['flag']:
			self.loader = self.pil_loader
		else:
			self.loader = kwargs['loader']

		self.get_all_names_refresh()


	def get_all_names_refresh(self):

		if os.path.exists(self.root+'/pickled_data_'+self.Type+'.p'):

			with open(self.root+'/pickled_data_'+self.Type+'.p', 'rb') as f:
				data_in = pickle.load(f)

			self.original = data_in['original']
			self.masked = data_in['masked']
			self.distribution = np.zeros([3, 2])
			self.paths = data_in['paths']

			for no, type_i in enumerate(self.paths):

				type__ = type_i.split('-')[0]

				if type__ == 'nose':

					self.distribution[0, 0] += np.sum(self.masked[no][:, :, 0])
					self.distribution[0, 1] += self.masked[no][:, :, 0].shape[0]*self.masked[no][:, :, 0].shape[1] - np.sum(self.masked[no][:, :, 0])

					self.distribution[1, 0] += np.sum(self.masked[no][:, :, 1])
					self.distribution[1, 1] += self.masked[no][:, :, 1].shape[0]*self.masked[no][:, :, 1].shape[1] - np.sum(self.masked[no][:, :, 1])

				elif type__ == 'cheek':

					self.distribution[0, 0] += np.sum(self.masked[no][:, :, 0])
					self.distribution[0, 1] += self.masked[no][:, :, 0].shape[0]*self.masked[no][:, :, 0].shape[1] - np.sum(self.masked[no][:, :, 0])

					self.distribution[1, 0] += np.sum(self.masked[no][:, :, 1])
					self.distribution[1, 1] += self.masked[no][:, :, 1].shape[0]*self.masked[no][:, :, 1].shape[1] - np.sum(self.masked[no][:, :, 1])

					self.distribution[2, 0] += np.sum(self.masked[no][:, :, 2])
					self.distribution[2, 1] += self.masked[no][:, :, 2].shape[0]*self.masked[no][:, :, 2].shape[1] - np.sum(self.masked[no][:, :, 2])

				elif type__ == 'forehead':

					self.distribution[0, 0] += np.sum(self.masked[no][:, :, 0])
					self.distribution[0, 1] += self.masked[no][:, :, 0].shape[0]*self.masked[no][:, :, 0].shape[1] - np.sum(self.masked[no][:, :, 0])

					self.distribution[2, 0] += np.sum(self.masked[no][:, :, 2])
					self.distribution[2, 1] += self.masked[no][:, :, 2].shape[0]*self.masked[no][:, :, 2].shape[1] - np.sum(self.masked[no][:, :, 2])

				elif type__ == 'chin':

					self.distribution[0, 0] += np.sum(self.masked[no][:, :, 0])
					self.distribution[0, 1] += self.masked[no][:, :, 0].shape[0]*self.masked[no][:, :, 0].shape[1] - np.sum(self.masked[no][:, :, 0])

					self.distribution[1, 0] += np.sum(self.masked[no][:, :, 1])
					self.distribution[1, 1] += self.masked[no][:, :, 1].shape[0]*self.masked[no][:, :, 1].shape[1] - np.sum(self.masked[no][:, :, 1])

			for i in range(self.distribution.shape[0]):
				self.distribution[i] = 1/self.distribution[i]
				self.distribution[i] = self.distribution[i]/np.sum(self.distribution[i])

			log.info("Succesfully loaded data from pickle file")

		else:
			log.info("Gandu "+self.Type +" ka basic pickle dump kaha hai?")


		if self.Type != 'test_one':

			self.what_is_annotated = json.load(open(self.root+'/flag.json', 'r'))

		if self.Type != 'train':

			if os.path.exists(self.root+'/pickled_data_extra'+self.Type+'.p'):

				with open(self.root+'/pickled_data_extra'+self.Type+'.p', 'rb') as f:
					self.main_data = pickle.load(f)
					# print(self.main_data)
				self.keys = sorted(list(self.main_data.keys()))

			else:
				log.info("Gandu "+self.Type +" ka extra pickle dump kaha hai?")	

		if self.Type != 'train':

			self.start = 0

	def show_img(self, img):

		plt.imshow(img)
		plt.show()

	def pil_loader(self, path):

		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def convert_float_tensor(self, x, y):

		x = torch.FloatTensor(x.transpose(0, 3, 1, 2))
		y = torch.FloatTensor(y.transpose(0, 3, 1, 2))

		return x, y

	def convert_float_tensor_one(self, x):

		x = torch.FloatTensor(x.transpose(0, 3, 1, 2))

		return x

	def __getitem__(self, index):

		if index>=self.__len__():
			raise IndexError

		if self.Type == 'train':
			random_images = np.random.choice(len(self.original), self.batchsize)
			img_p, target, path, what_is_annotated = [], [], [], []
			for i in random_images:
				img_p.append(self.original[i]/255)
				target.append(self.masked[i])
				path.append(self.paths[i])
				if self.Type != 'test_one':
					what_is_annotated.append(self.what_is_annotated['-'.join(self.paths[i].split('-')[2:])])

			cropped_img_p, cropped_target = self.convert_float_tensor(np.array(img_p), np.array(target))
			return cropped_img_p, cropped_target, path, what_is_annotated

		else:
			#Structure of main_data: {'name_img':{'coordinates':{..., original:bbox}, 'actual_image':image,target':{'cheek':..}, masks':{'cheek':..}}}

			# data has structure - {'cheek': torch.FloatTensor([3, 768, 768]),  ..., 'co_ordinates':{'cheek': [x, y, w, h], ..., 'original' : [height, width, 3]}}
			# target has structure - {'cheek': torch.FloatTensor([3, 768, 768]),  ...}
			# mask has structure - {'cheek': np.array([512, 512]),  ...} range(0, 1)

			#Return: no, (data, mask, target, path)

			# print("Here",self.Type)
			
			image_name = self.keys[index]
			data = {'co_ordinates': self.main_data[image_name]['coordinates']}
			target = {}

			for segment, crop in self.main_data[image_name]['target'].items():
				target[segment] = crop/255

			mask = self.main_data[image_name]['masks']

			for segment, crop in self.main_data[image_name]['masks'].items():
				self.main_data[image_name]['masks'][segment] = np.array(crop)/255

			for segment, crop in self.main_data[image_name]['actual_image'].items():
				data[segment] = self.convert_float_tensor_one(np.array(crop)[None, :, :, :]/255)

			if self.Type != 'test_one':

				return data, mask, target, image_name, self.what_is_annotated[image_name+'.png']

			else:

				return data, mask, target, image_name

	def __len__(self):

		if self.Type != 'train':

			return len(self.main_data.keys())
		
		else:

			return len(self.original)//self.batchsize