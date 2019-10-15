import yaml
import os
from .dl_model import dl_model
from .read_yaml import read_yaml
from .logger import Logger
#from .prepare_metadata.metadata import metadata

log = Logger()

class PipelineManager():

	def __init__(self):
		self.config_file = read_yaml()
		for i in self.config_file['dir']:
			if not os.path.exists(self.config_file['dir'][i]):
				os.mkdir(self.config_file['dir'][i])		
		
	#def prepare_metadata(self, mode):
	#	prepare_metadata(self.config_file, mode)

	def train(self, pipeline_name, model_name):
		train(pipeline_name, model_name)

	def test(self, pipeline_name, model_name):
		test(pipeline_name, model_name)

	# def ensemble(self, pipeline_name, model_name):
	# 	ensemble(pipeline_name, model_name)

	def test_one(self, pipeline_name, model_name):
		test_one_image(pipeline_name, model_name)

# def prepare_metadata(config_file, mode):
		
# 	if mode == 'both':	
# 		meta = metadata(config_file, 'train')
# 		meta.masks_generate()
# 		meta.generate_face_crops()
# 		meta.generate_crops()
# 		meta.load_save()
# 		meta2 = metadata(config_file, 'test')
# 		meta2.masks_generate()
# 		meta2.generate_face_crops()
# 		meta2.generate_crops()
# 		meta2.load_save()
# 	elif mode == 'train':
# 		meta = metadata(config_file, 'train')
# 		meta.masks_generate()
# 		meta.generate_face_crops()
# 		meta.generate_crops()
# 		meta.load_save()
# 	elif mode == 'test':
# 		meta = metadata(config_file, 'test')
# 		meta.masks_generate()
# 		meta.generate_face_crops()
# 		meta.generate_crops()
# 		meta.load_save()
# 	else:
# 		log.info(pipeline_name, 'Expected train or test')

def train(pipeline_name, model_name):
	
	# if pipeline_name == 'classification':
	# 	implemented = ['ResNet', 'AlexNet', 'InceptionNet']
	# 	if model_name in implemented :
	# 		driver = dl_model(model_name)
	# 		driver.train_model()
	# 	else:
	# 		log.info(model_name, ' Model not yet implemented')
	# elif pipeline_name == 'segmentation':
	# 	if model_name == 'UNet':
	# 		driver = dl_model(model_name)
	# 		driver.train_model()
	# 	elif model_name == 'ResNet_UNet':
	# 		driver = dl_model(model_name)
	# 		driver.train_model()
	# 	else:
	# 		log.info("Not yet implemented")
	if pipeline_name=='watermark_removal'
		if model_name=='UNet_watermark'
			driver = dl_model(model_name)
			driver.train_model()
		else:
			log.info(model_name, ' Model not yet implemented')
	elif pipeline_name=='text_highlighter'
		if model_name=='UNet_text'
			driver=dl_model(model_name)
			driver.train_model()
		else:
			log.info(model_name, ' Model not yet implemented')
	else:
		log.info(pipeline_name, ' Pipeline not yet implemented')

def test(pipeline_name, model_name):
	# if pipeline_name == 'classification':
	# 	implemented = ['ResNet', 'AlexNet', 'InceptionNet']
	# 	if model_name in implemented :
	# 		driver = dl_model(model_name, Training_Testing='Testing')
	# 		# driver.model.load(self.config_file['PreTrained_model']['checkpoint_best'])
	# 		driver.test_model()
	# 	else:
	# 		log.info(model_name, ' Model not yet implemented')
	# elif pipeline_name == 'segmentation':
	# 	if model_name == 'UNet':
	# 		driver = dl_model(model_name, Training_Testing='Testing')
	# 		driver.test_model()
	# 	elif model_name == 'ResNet_UNet':
	# 		driver = dl_model(model_name, Training_Testing='Testing')
	# 		driver.test_model()
	# 	else:
	# 		print("Not yet implemented")
	if pipeline_name == 'watermark_removal':
		if model_name=='UNet_watermark' :
			driver = dl_model(model_name, Training_Testing='Testing')
			# driver.model.load(self.config_file['PreTrained_model']['checkpoint_best'])
			driver.test_model()
		else:
			log.info(model_name, ' Model not yet implemented')
	elif pipeline_name == 'text_highlighter':
		if model_name == 'UNet_text':
			driver = dl_model(model_name, Training_Testing='Testing')
			driver.test_model()
		# elif model_name == 'ResNet_UNet':
		# 	driver = dl_model(model_name, Training_Testing='Testing')
		# 	driver.test_model()
		else:
			log.info(model_name, ' Model not yet implemented')

	else:
		log.info(pipeline_name, ' Pipeline not yet implemented')

# def ensemble(pipeline_name, model_name):

# 	if pipeline_name == 'classification':
# 		implemented = ['ResNet', 'AlexNet', 'InceptionNet']
# 		if model_name in implemented :
# 			driver = dl_model(model_name, Training_Testing='ensemble')
# 			# driver.model.load(self.config_file['PreTrained_model']['checkpoint_best'])
# 			driver.ensemble()
# 		else:
# 			log.info(model_name, ' Model not yet implemented')
# 	elif pipeline_name == 'segmentation':
# 		if model_name == 'UNet':
# 			driver = dl_model(model_name, Training_Testing='ensemble')
# 			driver.ensemble()
# 		elif model_name == 'ResNet_UNet':
# 			driver = dl_model(model_name, Training_Testing='ensemble')
# 			driver.ensemble()
# 		else:
# 			print("Not yet implemented")
# 	else:
# 		log.info(pipeline_name, ' Pipeline not yet implemented')


def test_one_image(pipeline_name, model_name):
	# if pipeline_name == 'classification':
	# 	implemented = ['ResNet', 'AlexNet', 'InceptionNet']
	# 	if model_name in implemented :
	# 		driver = dl_model(model_name, Training_Testing='test_one')
	# 		driver.test_one_image()
	# 	else:
	# 		log.info(model_name, ' Model not yet implemented')
	# elif pipeline_name == 'segmentation':
	# 	if model_name == 'UNet':
	# 		driver = dl_model(model_name, Training_Testing='test_one')
	# 		driver.test_one_image()
	# 	elif model_name == 'ResNet_UNet':
	# 		driver = dl_model(model_name, Training_Testing='test_one')
	# 		driver.test_one_image()
	# 	else:
	# 		print("Not yet implemented")
	# else:
	# 	log.info(pipeline_name, ' Pipeline not yet implemented')
	if pipeline_name == 'watermark_removal':
		if model_name=='UNet_watermark':
			driver = dl_model(model_name, Training_Testing='test_one')
			# driver.model.load(self.config_file['PreTrained_model']['checkpoint_best'])
			driver.test_one_image()
		else:
			log.info(model_name, ' Model not yet implemented')
	elif pipeline_name == 'text_highlighter':
		if model_name == 'UNet_text':
			driver = dl_model(model_name, Training_Testing='test_one')
			driver.test_one_image()
		# elif model_name == 'ResNet_UNet':
		# 	driver = dl_model(model_name, Training_Testing='Testing')
		# 	driver.test_model()
		else:
			log.info(model_name, ' Model not yet implemented')

	else:
		log.info(pipeline_name, ' Pipeline not yet implemented')

