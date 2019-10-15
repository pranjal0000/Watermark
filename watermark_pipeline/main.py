import click
from src.pipeline_manager import PipelineManager

from src.logger import Logger

@click.group()
def main():
	pass

#@main.command()
#@click.option('-m', '--mode', help='train/test?', required=True)
# @click.option('-m', '--model', help='ResNet, IncepNet, AlexNet' , required=True)
#def prepare_metadata(mode):
#    pipeline_manager.prepare_metadata(mode)

@main.command()
@click.option('-p', '--pipeline_name', help='watermark_removal,text_highlighter', required=True)
@click.option('-m', '--model', help='UNet_watermark,UNet_text' , required=True)
def train(pipeline_name, model):
    pipeline_manager.train(pipeline_name, model)

@main.command()
@click.option('-p', '--pipeline_name', help='watermark_removal,text_highlighter', required=True)	
@click.option('-m', '--model', help='UNet_watermark,UNet_text' , required=True)
def test(pipeline_name, model):
    pipeline_manager.test(pipeline_name, model)

# @main.command()
# @click.option('-p', '--pipeline_name', help='classification,segmentation', required=True)	
# @click.option('-m', '--model', help='ResNet, IncepNet, AlexNet,UNet,ResNet_UNet' , required=True)
# def ensemble(pipeline_name, model):
#     pipeline_manager.ensemble(pipeline_name, model)

@main.command()
@click.option('-p', '--pipeline_name', help='watermark_removal,text_highlighter', required=True)
@click.option('-m', '--model', help='UNet_watermark,UNet_text' , required=True)
def test_one(pipeline_name, model):
    pipeline_manager.test_one(pipeline_name, model)


if __name__ == "__main__":

	pipeline_manager = PipelineManager()
	log = Logger()
	log.first()
	
	main()
