import tkinter as tk
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
class Paint:
    def __init__(self):
        self.root = tk.Tk()
        self.canvas_width = 200
        self.canvas_height = 200
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.paint)
        self.button_submit = tk.Button(self.root, text='Submit', command=self.submit)
        self.button_submit.pack(side='left')
        self.button_reset = tk.Button(self.root, text='Reset', command=self.reset)
        self.button_reset.pack(side='left')
        self.image = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)

    def paint(self, event):
        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.image[y1:y2, x1:x2] = 255

    def submit(self):
        save_image(self.image,"image.jpg")
        img_arr = load_image("image.jpg")
        kernel = np.array([[1/16, 2/16, 1/16],
                   [2/16, 4/16, 2/16],
                   [1/16, 2/16, 1/16]])
        blur_img =convolve2d(img_arr,kernel,mode='same')
        show_image(blur_img)
    def reset(self):
        self.canvas.delete('all')
        self.image.fill(0)
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = np.array(img)
    return img
def save_image(img,path):
    cv2.imwrite(path, img)
def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()
paint = Paint()
paint.root.mainloop()
