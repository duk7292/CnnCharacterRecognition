import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import utils as ut
import matplotlib.pyplot as plt
import tkinter as tk
import cv2

from scipy.signal import convolve2d

learning_rate = 0.00005

class Paint:
    def __init__(self,Layers):
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
        self.layers = Layers
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
        ut.CNN_forward(blur_img, self.layers)[0]
        output = ut.CNN_forward(blur_img, self.layers)[0]
        print(np.argmax(output))
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


def main():
    # CSV-Datei einlesen
    dataRaw = pd.read_csv('balanced_data.csv').to_numpy().astype(float)
    np.random.shuffle(dataRaw)
    
    data_Normalized = ut.normalizeData(dataRaw)
    data_Normalized = np.array(data_Normalized)

    labels = data_Normalized[:, 0].astype(int)
    images_1d = data_Normalized[:, 1:]
    
    image_size = 28
    images_2d = images_1d.reshape(-1, image_size, image_size)

    labels_train, labels_test, images_train, images_test = train_test_split(labels, images_2d, test_size=0.1)
 

    

    layers = [
        ut.ConvolutionLayer(32,3,learning_rate),
        ut.ReLuLayer(),
        ut.MaxPooling2D(2),
        ut.ConvolutionLayer(32,3,learning_rate),
        ut.ReLuLayer(),
        ut.MaxPooling2D(2),
        ut.FlattenLayer(),
        ut.DenseLayer(126,50176,learning_rate),
        ut.ReLuLayer(),
        ut.DenseLayer(26,126,learning_rate),
        ut.softmaxLayer()
            ]
    def train_in_batches(images_train, labels_train, layers, batch_size=1000):
        global learning_rate
       
        num_batches = len(images_train) // batch_size
        accuracies = []
        print(num_batches)
        for batch_idx in range(num_batches):
            
            batch_predictions = []
            batch_labels = []
            batch_label_arr = []
         
            if(batch_idx % 30 == 0):
                learning_rate /=5
                print(learning_rate)

         
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            last_label = 0
            total_batch_label_arr = np.zeros(26)
            total_batch_predictions = np.zeros(26)
            for i, (image, label) in enumerate(zip(images_train[start_idx:end_idx], labels_train[start_idx:end_idx])):
                output = ut.CNN_forward(image, layers)[0]
                
                label_arr = np.zeros(len(output))
                label_arr[label] = 1
                
                
                batch_predictions.append(output)
                batch_labels.append(label)
                batch_label_arr.append(label_arr)
                total_batch_label_arr += label_arr
                total_batch_predictions += output
            error =  total_batch_label_arr - total_batch_predictions
            ut.CNN_backward(error, layers)
            
            for i in range(len(batch_labels)):

                print(batch_labels[i], np.argmax(batch_predictions[i]))
           
            batch_predictions = np.array(batch_predictions)
            batch_labels = np.array(batch_labels)

            print((batch_predictions[0]),batch_labels[0],np.argmax(batch_predictions[0]))

        
            accuracy = ut.calculate_accuracy(batch_predictions, batch_labels)
            accuracies.append(accuracy)
            print(f"Genauigkeit für Batch {batch_idx + 1}: {accuracy:.2f}%\nDurchschnittliche Genauigkeit über alle Batches: { np.mean(accuracies):.2f}%\nHöste Genauigkeit bis Jetzt {accuracies[np.argmax(accuracies)]:.2f}% bei batch {np.argmax(accuracies)}")
            if(accuracy > 97):
                return
        return np.mean(accuracies)

    train_in_batches(images_train, labels_train, layers)
    
    paint = Paint(layers)
    paint.root.mainloop()   
        
        
        

if __name__ == "__main__":
    main()