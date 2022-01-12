from island_predictor import LabelledImage, IslandPredictor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import os

if __name__ == '__main__':
	model = IslandPredictor()
	model.precache_features()

	path = Path(os.path.dirname(os.path.abspath(__file__)))
	map_path = path / '..' / 'testing-and-validation' / 'named-treasure-maps'
	map_image_paths = map_path.glob('**/*.jpg')

	x_ims = []
	for path in map_image_paths:
		x_ims.append(LabelledImage(str(path)))

	preds = [model.predict(im.im) for im in x_ims]
	true = [im.name for im in x_ims]
	cm = confusion_matrix(true, preds)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot()
	plt.show()
	plt.waitforbuttonpress()