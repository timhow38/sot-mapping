import cv2
from skimage import exposure
import numpy as np

def apply_clahe(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE()
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def match_histograms(source, reference):
    matched = exposure.match_histograms(source, reference, channel_axis=-1)
    return matched.astype('uint8')

def apply_mask(im, mask):
    return cv2.bitwise_and(im, im, mask=mask)

def clip_square_section(im, x, y, w, h):
    section = im.copy()[y:y+h, x:x+w]
    largest_axis = np.max(section.shape)
    bottom_padding = largest_axis - section.shape[0]
    right_padding = largest_axis - section.shape[1]
    section = cv2.copyMakeBorder(section, 0, bottom_padding, 0, right_padding, cv2.BORDER_CONSTANT, value=0)
    return section

def resize(im, target_size=400):
    if type(target_size) == int:
        target_size = (target_size, target_size)
    return cv2.resize(im, target_size)

def extract_colour_histogram(im):
    hist_b = cv2.calcHist([im], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([im], [0], None, [256], [0,256])
    hist_r = cv2.calcHist([im], [0], None, [256], [0,256])
    pass
