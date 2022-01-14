import tkinter as tk
import win32gui
import win32con
from win32api import GetSystemMetrics
from PIL import ImageTk,Image
import threading

# https://github.com/notatallshaw/fall_guys_ping_estimate/blob/main/fgpe/overlay.py
class UiOverlayManager:
    """
    Creates an overlay window using tkinter
    Uses the "-topmost" property to always stay on top of other Windows
    """
    def __init__(self, 
        initial_text,
        confirm_callback,
        reject_callback,
        timeout_callback):

        self.initial_text = initial_text
        self.confirm_callback = confirm_callback
        self.reject_callback = reject_callback
        self.timeout_callback = timeout_callback
        self.root = tk.Tk()
        self._job = None
        hwnd = win32gui.FindWindow(None, 'Sea of Thieves')
        window_y_offset = 40
        x = 0
        y = -window_y_offset
        w = GetSystemMetrics(0)
        h = GetSystemMetrics(1) + window_y_offset
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, win32con.WS_POPUP)
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, x, y, w, h, 0)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        taskbar_hwnd = win32gui.FindWindow('Shell_traywnd', None)
        win32gui.ShowWindow(taskbar_hwnd, win32con.SW_HIDE)

        # Set up close label
        self.close_label = tk.Label(
            self.root,
            text=' X |',
            font=('Consolas', '14'),
            fg='green3',
            bg='grey19'
        )
        self.close_label.bind("<Button-1>", self._on_close)
        self.close_label.grid(row=0, column=0)

        # Set up content label
        self.content_text = tk.StringVar()
        self.content_label = tk.Label(
            self.root,
            textvariable=self.content_text,
            font=('Consolas', '14'),
            fg='green3',
            bg='grey19'
        )
        self.content_label.grid(row=0, column=1)

        # Set up prediction canvas
        self.predicted_image = tk.Canvas(self.root, width=0, height=0)
        self.predicted_image.grid(row=1, column=0, columnspan=2)

        # Set up confirm/reject buttons
        self.confirm_label = tk.Label(
            self.root,
            text=' Y |',
            font=('Consolas', '14'),
            fg='green3',
            bg='grey19'
        )
        self.confirm_label.bind("<Button-1>", lambda *args: self.reset_data_then(0,confirm_callback))
        self.confirm_label.grid(row=2, column=0)

        self.reject_label = tk.Label(
            self.root,
            text=' N',
            font=('Consolas', '14'),
            fg='green3',
            bg='grey19'
        )
        self.reject_label.bind("<Button-1>", lambda *args: self.reset_data_then(0,reject_callback))
        self.reject_label.grid(row=2, column=1)

        self.confirm_label.grid_remove()
        self.reject_label.grid_remove()

        # Define Window Geometery
        self.root.overrideredirect(True)
        #self.root.geometry("+5+5")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.lift()
        self.root.wm_attributes("-topmost", True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._on_close()

    def _on_close(self, *args):
        self.root.destroy()
        taskbar_hwnd = win32gui.FindWindow('Shell_traywnd', None)
        win32gui.ShowWindow(taskbar_hwnd, win32con.SW_SHOW)

    def queue_action(self, wait_time_seconds, action):
        if self._job is not None:
            self.root.after_cancel(self._job)
        if action is not None:
            self._job = self.root.after(wait_time_seconds*1000, action)

    def reset_data(self, trigger_callback=True):
        self.content_text.set('Ready')
        self.predicted_image.delete('all')
        self.predicted_image.config(width=0, height=0)
        self.confirm_label.grid_remove()
        self.reject_label.grid_remove()
        if trigger_callback:
            self.timeout_callback()

    def reset_data_then(self, wait_time_seconds=0, action=None):
        self.reset_data(False)
        self.queue_action(wait_time_seconds, action)

    def update_label(self, text):
        self.content_text.set(text)

    def update_label_then(self, text, wait_time_seconds=10, next_update=None):
        if next_update is None:
            next_update = self.reset_data
        self.update_label(text)
        self.queue_action(wait_time_seconds, next_update)

    def update_image(self, im_arr):
        self.predicted_image_data = ImageTk.PhotoImage(image=Image.fromarray(im_arr))
        self.predicted_image.config(width=400, height=400)
        self.predicted_image.create_image(400,20,anchor='ne',image=self.predicted_image_data)
        self.confirm_label.grid()
        self.reject_label.grid()

    def update_image_then(self, im_arr, wait_time_seconds=10, next_update=None):
        if next_update is None:
            next_update = self.reset_data
        self.update_image(im_arr)
        self.queue_action(wait_time_seconds, next_update)

    def run(self):
        self.content_text.set(self.initial_text)
        self.root.mainloop()
        

if __name__ == "__main__":
    overlay = UiOverlayManager()
    overlay.run()
