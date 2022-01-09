from extract_mask import extract_mask
from pathlib import Path
import cv2
import numpy as np

if __name__ == '__main__':
    # TODO relative paths
    current_dir = r'C:\Users\Riley\source\repos\SOT\sot-mapping'
    path = Path(current_dir)
    path = path / 'mapAsset-sources' / 'assets-loc-img'
    image_paths = path.glob('**/*.jpg')

    write_dir = r'C:\Users\Riley\source\repos\SOT\sot-mapping\training-images'
    for (index, path) in enumerate(list(image_paths)):
        name = str(path).split('-')[-1].split('.')[0]
        img = cv2.imread(str(path))
        mask = extract_mask(img).astype('uint8')
        cv2.imwrite(write_dir + '\\' + name + '.png', mask)