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

from imutils import face_utils
import imutils
from .data_loader import ownDataLoader
from .dl_model import dl_model

class ensemble_model():

	def __init__(self, model_paths, pickle_path, mode, config):

		self.config = config
		self.root = config[mode]['dir']

#Define the model, preprocessing step for each model
class single_model():

	def __init__(name, path_to_model):

		self.model = dl_model(name)
		self.model

	def preprocess_before_out(self):





class data_loader():

	def __init__(self, mode, config):

		self.config = config
		self.root = config[mode]['dir']

		self.load_data()

	def load_data():

		if os.path.exists(self.root+'/pickled_data_'+mode+'.p'):

			with open(self.root+'/pickled_data_'+self.Type+'.p', 'rb') as f:
				data_in = pickle.load(f)

			self.original = data_in['original']
			self.masked = data_in['masked']
			self.paths = data_in['paths']
			self.distribution = data_in['distribution']

			log.info("Succesfully loaded data from pickle file")

		else:
			print("gandu pickle dump kaha hai?")		

		def __getitem__(self, index):

		def __len__(self):

			return len(self.original)