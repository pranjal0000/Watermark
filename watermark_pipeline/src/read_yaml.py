import yaml
def read_yaml():
	with open("configs/config.yaml", 'r') as stream:
		try:
			return yaml.load(stream)
		except yaml.YAMLError as exc:
			return exc