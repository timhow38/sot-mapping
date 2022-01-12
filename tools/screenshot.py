from pynput import keyboard
from extract_mask import extract_mask
from feature_extractor import extract_hu_moments
from PIL import ImageGrab
import numpy as np
import cv2

def on_activate():
    im = np.array(ImageGrab.grab().convert('RGB'))
    mask = extract_mask(im)
    cv2.imshow('Raw', im)
    cv2.imshow('Central contours', mask)
    moments = extract_hu_moments(mask)
    print(moments)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def quit():
    return False

if __name__ == '__main__':
    with keyboard.GlobalHotKeys({
        '<ctrl>+<shift>+m': on_activate,
        '<ctrl>+c': quit
        }) as h:
        h.join()