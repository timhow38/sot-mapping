from pynput import keyboard
from island_predictor import IslandPredictor
from PIL import ImageGrab
import numpy as np
import cv2
import uuid
import time
import threading
import os
from user_interface import UiOverlayManager
import shelve

def on_activate(model, ui):
    global last_prediction
    ui.update_label('Capturing...')
    print('Pausing for map lift')
    time.sleep(2)
    print('Predicting...')
    im = np.array(ImageGrab.grab().convert('RGB'))
    ui.update_label('Capture complete! Predicting...')
    prediction, prediction_image = model.predict(im)
    if prediction is not None:
        last_prediction = [prediction, im]
        ui.update_label_then(prediction)
        ui.update_image_then(prediction_image)
    else:
        ui.update_label_then('No prediction, model not ready')

def on_press(key, model, ui):
    global current
    current.add(key)
    if all(k in current for k in capture_chord):
        threading.Thread(target=on_activate, args=(model,ui)).start()

def on_release(key):
    global current
    if key in current:
        current.remove(key)

def save_image(prediction, im, path):
    fullpath = path + f'maybe_{prediction}_{str(uuid.uuid4())[:9]}.jpg'
    print(f'Saving as {fullpath}')
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(fullpath, im)

def save_last_prediction(path):
    global last_prediction
    if last_prediction != []:
        save_image(*last_prediction, path)
        last_prediction = []

def confirm_callback(*args, rootpath=r'C:\temp\sot-captures\\'):
    path = rootpath + r'confirmed\\'
    save_last_prediction(path)

def reject_callback(*args, rootpath=r'C:\temp\sot-captures\\'):
    path = rootpath + r'rejected\\'
    save_last_prediction(path)

def timeout_callback(*args, rootpath=r'C:\temp\sot-captures\\'):
    path = rootpath + r'unknown\\'
    save_last_prediction(path)

if __name__ == '__main__':
    current = set()
    model = IslandPredictor()
    capture_chord = set([keyboard.Key.space, keyboard.KeyCode(char='q'),  keyboard.KeyCode(char='e')])
    last_prediction = []
    with UiOverlayManager(
        'Ready',
        confirm_callback,
        reject_callback,
        timeout_callback) as ui:

        with keyboard.Listener(
            on_press=lambda *args: on_press(*args, model, ui), 
            on_release=on_release) as listener:

            model.precache_features()
            threading.Thread(target=listener.join).start()
            ui.run()