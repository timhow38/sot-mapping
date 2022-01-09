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
    big_concat = None
    row = None
    width = 5
    num_items = 10
    for (index, path) in enumerate(list(image_paths)[:num_items]):
        name = str(path).split('-')[-1]
        img = cv2.imread(str(path))
        mask = extract_mask(img)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype('uint8')
        concat = cv2.vconcat([img, mask])
        concat = cv2.resize(concat, (320,640))
        if row is None:
            row = concat
        else:
            row = cv2.hconcat([row, concat])
            if (index + 1) % width == 0:
                if big_concat is None:
                    big_concat = row
                else:
                    big_concat = cv2.vconcat([big_concat, row])
                row = None
            if index == num_items - 1 and row is not None:
                white = np.zeros((640, 320, 3)).astype('uint8')
                for i in range((index + 1) % width):
                    row = cv2.hconcat([row, white])
                big_concat = cv2.vconcat([big_concat, row])

    cv2.imshow(name, big_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()