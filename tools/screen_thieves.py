from pynput import keyboard
from island_predictor import IslandPredictor
from PIL import ImageGrab
import numpy as np
import cv2
import uuid
import time
import threading
import os

def on_activate(model):
    current = set()
    print('Pausing for map lift')
    time.sleep(2)
    print('Predicting...')
    im = np.array(ImageGrab.grab().convert('RGB'))
    prediction = model.predict(im)
    if prediction is not None:
        print('Saving as ')
        print(r'C:\temp\sot-captures\\' + f'maybe_{prediction}_{str(uuid.uuid4())[:9]}.jpg')
        if not os.path.exists(r'C:\temp'):
            os.mkdir(r'C:\temp')
        if not os.path.exists(r'C:\temp\sot-captures'):
            os.mkdir(r'C:\temp\sot-captures')
        cv2.imwrite(r'C:\temp\sot-captures\\' + f'maybe_{prediction}_{str(uuid.uuid4())[:9]}.jpg', im)
    
def parse_key(key):
    if type(key) == keyboard.Key:
        return key.value
    return key

def quit():
    print('cya')
    return False

def on_press(key): 
    current.add(key)
    if all(k in current for k in capture_chord):
        threading.Thread(target=on_activate, args=(model,)).start()
    if all([k in current for k in quit_chord]): 
        return False

def on_release(key):
    if key in current:
        current.remove(key)

if __name__ == '__main__':
    current = set()
    model = IslandPredictor()
    capture_chord = set([keyboard.Key.space, keyboard.KeyCode(char='q')])
    quit_chord = set([keyboard.Key.shift, keyboard.KeyCode(char='a')])
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        model.precache_features()
        listener.join()
    