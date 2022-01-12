import cv2
import numpy as np
from pathlib import Path
import os
import pandas as pd
import re
from feature_extractor import get_features
from matching_algorithm import match_image

class LabelledImage:
	def __init__(self, path_str):
		self.path = path_str
		self.im = cv2.imread(path_str)
		self.set_info_from_path(path_str)

	def set_info_from_path(self, path):
		self.file_name = path.split('\\')[-1]
		info_vals = re.split('-_', self.file_name)
		if len(info_vals) == 4:
			self.area = info_vals[1]
			self.grid_square = info_vals[2]
			self.name = info_vals[3].split('.')[0]
		else:
			self.name = info_vals[0]

	def set_features(self):
		self.features = get_features(self.im)

class IslandPredictor:
	def __init__(self):
		self.match_image = match_image
		self.cache = {}
		self.ref_ims = None

	def get_reference_ims(self):
		path = Path(os.path.dirname(os.path.abspath(__file__)))
		path = path / '..' / 'assets-mapData' / 'assets-loc-img'
		# TODO figure out actual pattern matching
		# lots of shitty wrong answers on SO
		image_paths = list(path.glob('**/*.*p*'))
		self.ref_ims = []
		for path in image_paths:
			self.ref_ims.append(LabelledImage(str(path)))

	def precache_features(self):
		print(f'Precaching model features...')
		if self.ref_ims is None:
			print(f'Loading reference images...')
			self.get_reference_ims()
			print(f'Reference images loaded')
		for image in self.ref_ims:
			image.set_features()
			self.cache[image.file_name] = image.features
		print(f'Precaching complete')

	def predict(self, im, full_predictions=False):
		predictions = self.match_image(im, self.cache)
		predictions = sorted(predictions.items(), key=lambda x: x[1][0])
		if full_predictions:
			return predictions
		else:
			return predictions[0][0]
