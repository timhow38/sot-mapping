from island_predictor import LabelledImage, IslandPredictor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import os

from feature_extractor import get_features
import cv2

if __name__ == '__main__':
	model = IslandPredictor()
	model.precache_features()

	path = Path(os.path.dirname(os.path.abspath(__file__)))
	map_path = path / '..' / 'testing-and-validation' / 'named-treasure-maps'
	map_image_paths = map_path.glob('**/*.*p*g*')

	x_ims = []
	for path in map_image_paths:
		x_ims.append(LabelledImage(str(path)))

	preds = [model.predict(im.im) for im in x_ims]
	preds = [pred.split('-')[-1].split('.')[0] for pred in preds]
	true = [im.name for im in x_ims]
	true = [name.split('_')[0] for name in true]
	for y_pred, y_true in zip(preds, true):
		print(f'{y_pred} -> {y_true} {y_pred == y_true}')
	cm = confusion_matrix(true, preds)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot()
	plt.show()
	plt.waitforbuttonpress()