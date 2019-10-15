import torch
import torch.nn as nn
import numpy as np
from ..logger import Logger

log = Logger()

class model(nn.Module):

	def accuracy(self, x, y):

		_, arg = torch.max(x, dim=1)

		eq = torch.eq(arg.squeeze(), y.squeeze())

		return torch.mean(eq.float())

		# print(self.config['dir']['Exp'])

	def print_info(self, info):

		log.info('The average accuracy is :', np.mean(info['Acc']))
		log.info('The current accuracy is :', info['Acc'][-1])
		log.info('The average loss is :', np.mean(info['Loss']))
		log.info('The current loss is :', info['Loss'][-1])

	def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar'):

		torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict()},self.config['dir']['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
		torch.save(info, self.config['dir']['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
		if is_best:
			
			shutil.copyfile(self.config['dir']['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
			shutil.copyfile(self.config['dir']['Exp']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

	def load(self, path, path_info):

		checkpoint = torch.load(self.config['dir']['Exp']+'/'+path)

		self.load_state_dict(checkpoint['state_dict'])
		self.opt.load_state_dict(checkpoint['optimizer'])

		return checkpoint['epoch'], torch.load(self.config['dir']['Exp']+'/'+path_info)