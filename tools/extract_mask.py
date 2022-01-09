from PIL import Image, ImageGrab
import numpy as np
import cv2
import sys

def extract_mask(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(opening, 50, 200)
    dilated = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center_y, center_x = gray.shape
    center_x /= 2
    center_y /= 2
    central_contours_mask = np.zeros(gray.shape)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        moments = cv2.moments(contour)
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        dx = abs(x - center_x)
        dy = abs(y - center_y)
        if dy <= center_y / 2 and dx <= center_x / 2:
            cv2.fillPoly(central_contours_mask, pts=[contour], color=(255,0,0))
    return central_contours_mask

if __name__ == '__main__':
    if len(sys.argv) > 1:
        im = np.array(Image.open(sys.argv[1]).convert('RGB'))
    else:
        im = np.array(ImageGrab.grab().convert('RGB'))
    mask = extract_mask(im)
    cv2.imshow('Raw', im)
    cv2.imshow('Central contours', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()