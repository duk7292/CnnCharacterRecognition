import utils as ut
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time


dataRaw = pd.read_csv('MainFiles\\balanced_data.csv').to_numpy().astype(float)
np.random.shuffle(dataRaw)

data_Normalized = ut.normalizeData(dataRaw)
data_Normalized = np.array(data_Normalized)

labels = data_Normalized[:, 0].astype(int)
images_1d = data_Normalized[:, 1:]

image_size = 28
images_2d = images_1d.reshape(-1, image_size, image_size)

labels_train, labels_test, images_train, images_test = train_test_split(labels, images_2d, test_size=0.1)
lrList = [5e-3,3e-3,1e-3,5e-4,3e-4,1e-4,5e-5,3e-5,1e-5,5e-6]
lr = lrList[0]






layers = [
    ut.ConvolutionLayer(6,3,lr),
    ut.MaxPooling2D(2),
    ut.ReLuLayer(),
    ut.ConvolutionLayer(16,3,lr),
    ut.MaxPooling2D(2),
    ut.FlattenLayer(),
    ut.DenseLayer(180,4704,lr),
    ut.ReLuLayer(),
    ut.DenseLayer(80,180,lr),
    ut.ReLuLayer(),
    ut.DenseLayer(26,80,lr),
    ut.SoftmaxLayer()
]

batch_predictions = []
batch_labels = []
accuracies = []

def test(test_size = 1000):
    correct = 0
    for i, (image, label) in enumerate(zip(images_test[:test_size], labels_test[:test_size])):
        output = np.argmax(ut.CNN_forward(image, layers))
        if (output == label):
            correct+=1
    print((correct/test_size)*100,"%")

def train(batch_size = 250):
    global lr
    global lrList
    num_batches = len(images_train) // batch_size
    accuracies = []
    for epoch in range(1000):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_predictions = []
            batch_labels = []
            layers[0].learning_rate = lr
            layers[3].learning_rate = lr
            layers[6].learning_rate = lr
            layers[8].learning_rate = lr
           # layers[10].learning_rate = lr
            t = time.time()
            for i, (image, label) in enumerate(zip(images_train[start_idx:end_idx], labels_train[start_idx:end_idx])):
                output = ut.CNN_forward(image, layers)
                label_arr = np.zeros(len(output))
                label_arr[label] = 1
                batch_predictions.append(output)
                batch_labels.append(label)
                error =(output- label_arr)
                ut.CNN_backward(error, layers)

            print(time.time()-t)
            accuracy = ut.calculate_accuracy(batch_predictions, batch_labels)
            accuracies.append(accuracy) 

            print(f"Genauigkeit für Batch {(epoch*num_batches)+batch_idx + 1}: {accuracy:.2f}%\nDurchschnittliche Genauigkeit über alle Batches: { np.mean(accuracies):.2f}%\nHöste Genauigkeit bis Jetzt {accuracies[np.argmax(accuracies)]:.2f}% bei batch {np.argmax(accuracies)+1}")
            if(accuracy >98.5 ):
                return
            
            learning_rate_idx = int(accuracy // 10)

            lr = lrList[learning_rate_idx]


        
train()
test()