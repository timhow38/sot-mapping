import cv2
from math import log, copysign
import numpy as np
from image_tools import resize, clip_square_section, apply_clahe, apply_mask, match_histograms
from pathlib import Path
import os

def extract_hu_moments(mask):
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments)
    log_moments = [log(abs(moment)) for moment in hu_moments]
    return np.array(log_moments)

def extract_central_contours(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3,3),np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closed, 50, 200)

    lines = cv2.HoughLinesP(cv2.dilate(edges, kernel),1,np.pi/2, 50, None, 50, 1)
    if lines is not None:
        lines_im = np.zeros(edges.shape).astype('uint8')
        for line in lines:
            x0,y0,x1,y1 = line[0]
            cv2.line(lines_im,(x0,y0),(x1,y1),255,4)
        edges = cv2.bitwise_and(edges, ~lines_im)

    final_edges = cv2.dilate(edges, kernel)
    close_iters = 2
    kernel = np.ones((5,5),np.uint8)
    for i in range(close_iters):
        final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(final_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center_y, center_x = gray.shape
    center_x /= 2
    center_y /= 2
    central_contours_mask = np.zeros(gray.shape).astype('uint8')
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
            cv2.fillPoly(central_contours_mask, pts=[contour], color=(255))
    return central_contours_mask

def extract_bound_mask(mask):
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours) > 0:
        return None, None
    contours = np.concatenate(contours)
    x,y,w,h = cv2.boundingRect(contours)
    return clip_square_section(mask,x,y,w,h), x,y,w,h

def extract_closest_colour_masks(im, colours, return_recoloured=False, using_rgb=True):
    if using_rgb:
        colours = np.flip(colours, axis=1)
    diffs = np.zeros((im.shape[0], im.shape[1], len(colours)))
    for i, colour in enumerate(colours):
        diffs[:,:,i] = np.sum(np.abs(im - colour), axis=-1)
    masks = diffs.argmin(axis=-1)
    if return_recoloured:
        out = np.zeros(im.shape)
        for i, colour in enumerate(colours):
            out[masks == i] = colour
        return out.astype('uint8')
    return masks.astype('uint8')

def extract_sand_grass_rock(im):
    # TODO store reference histo somewhere

    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    image_paths = list(path.glob('**/treasure3.jpg'))
    ref = cv2.imread(str(image_paths[0]))

    ref_mask, *ref_coords = extract_bound_mask(extract_central_contours(ref))
    ref = clip_square_section(ref, *ref_coords)
    ref = apply_clahe(ref)
    clipped_ref = resize(apply_mask(ref, ref_mask))

    clipped = resize(apply_clahe(im))

    im = match_histograms(clipped, clipped_ref)
    colours = [
        [0,0,0], # Bounds
        [243,255,255], # Sand
        [100, 160, 40], # Grass
        [132, 148, 130], # Rock
    ]
    colour_masks = extract_closest_colour_masks(clipped, colours, True)
    masks = []
    for i, colour in enumerate(colours):
        masks.append(np.all(colour_masks == colour, axis=-1).astype('uint8'))
    return masks