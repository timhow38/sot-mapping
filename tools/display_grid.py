import cv2
import numpy as np
from image_tools import resize

def create_display_grid(im_arr, resize_all=None):
	if len(im_arr) == 1:
		im_arr = [im_arr]
	for im in im_arr[0]:
		if type(im) == np.ndarray:
			shape = im.shape
			break
	if len(shape) == 2:
		shape = [*shape, 3]
	row_accumulator = None
	full_accumulator = None
	for row in im_arr:
		for im in row:
			if resize_all is not None:
				im = resize(im, resize_all)
			if type(im) == int and im == 0:
				im = np.zeros(shape)
			im = im.astype('uint8')
			if len(im.shape) == 2:
				im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
			if row_accumulator is None:
				row_accumulator = im
			else:
				row_accumulator = cv2.hconcat([row_accumulator, im])
		if full_accumulator is None:
			full_accumulator = row_accumulator
		else:
			full_accumulator = cv2.vconcat([full_accumulator, row_accumulator])
		row_accumulator = None
	return full_accumulator