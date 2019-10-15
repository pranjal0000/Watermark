from datetime import datetime
from .read_yaml import read_yaml
import sys

class Logger():

	def __init__(self):
		self.config = read_yaml()
		self.write_path = self.config['dir']['Exp']+'/log.txt'
		self.write_path_err = self.config['dir']['Exp']+'/log_err.txt'
		del self.config
		# sys.stderr = open(self.write_path_err, 'a')
		self.f = open(self.write_path, 'a')
		# self.g = open(self.write_path_err, 'a')

	def first(self):
		self.f.write('\n--------- Starting new session: '+ str(datetime.now().time()) +' ---------\n\n')
		# print('\n--------- Starting new session: '+ str(datetime.now().time())[:-7] +' ---------\n\n', file=sys.stderr)
	
	def info(self, *args):

		temp = ' '.join([str(i) for i in args])

		if "".join(temp.split()) == '\n':
			log_string = '\n'
			string = '\n'
		elif "".join(temp.split()) == '':
			string = ''
			log_string = ''
		else:
			string = str(datetime.now().time())[:-7] +': '+temp
			log_string = str(datetime.now().time()) +': '+temp
		print(string)
		self.f.write(log_string+'\n')