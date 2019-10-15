import numpy as np
import cv2
import os
import json
import pickle
import matplotlib.pyplot as plt

class generate_json():

	def __init__(self):

		self.mapping={0:'Spot',1:'Patch',2:'Wrinkle'}

		return

	def generate(self, annot, orig_image_path):

		data=[]

		data.append({"filename":orig_image_path, "class":"image", "annotations":[]})

		for i in range(3):

			_,contours, _ = cv2.findContours(np.array(annot[:,:,i]),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			
			for contour in contours:

				to_append = {"class":self.mapping[i]}

				xn = [str(x[0][0]) for x in contour]
				yn = [str(x[0][1]) for x in contour]

				xstr = ';'.join(xn)
				ystr = ";".join(yn)

				to_append["xn"] = xstr
				to_append["yn"] = ystr

				data[-1]["annotations"].append(to_append)

		return data

		
