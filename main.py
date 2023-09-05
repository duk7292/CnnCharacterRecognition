import utils as ut
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time


dataRaw = pd.read_csv('balanced_data.csv').to_numpy().astype(float)
np.random.shuffle(dataRaw)

data_Normalized = ut.normalizeData(dataRaw)
data_Normalized = np.array(data_Normalized)

labels = data_Normalized[:, 0].astype(int)
images_1d = data_Normalized[:, 1:]

image_size = 28
images_2d = images_1d.reshape(-1, image_size, image_size)

labels_train, labels_test, images_train, images_test = train_test_split(labels, images_2d, test_size=0.1)
lrList = [1e-3,1e-3,1e-3,1e-3,5e-4 ,5e-4,5e-4,1e-5,1e-5,5e-6]
lr = lrList[0]






layers = [
    ut.ConvolutionLayer(32,3,lr),
    ut.MaxPooling2D(2),
    ut.ReLuLayer(),
    ut.ConvolutionLayer(16,3,lr),
    ut.MaxPooling2D(2),
    ut.FlattenLayer(),
    ut.DenseLayer(1000,25088,lr),
    ut.ReLuLayer(),
    ut.DenseLayer(70,1000,lr),
    ut.ReLuLayer(),
    ut.DenseLayer(26,70,lr),
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
def train(batch_size = 250 ,back_cluster_size = 1):
    st = time.time()
    global lr
    global lrList
    outputSize = 26
    num_batches = (len(images_train) // batch_size) //back_cluster_size

    cluster_num = batch_size// back_cluster_size
    accuracies = []
    for epoch in range(1000000):
        for batch_idx in range(num_batches):
            if(batch_idx % 10 == 0):
                
                labels_guessed = np.zeros((26,2))
            batch_predictions = []
            batch_labels = []
            
            layers[0].setLR(lr)
            layers[3].setLR(lr)
            layers[6].setLR(lr)
            layers[8].setLR(lr)
            layers[10].setLR(lr)
            


            for cluster in range(cluster_num):
                start_idx = batch_idx * batch_size +(cluster*back_cluster_size)
                end_idx = start_idx + back_cluster_size
                cluster_error = np.zeros((back_cluster_size,outputSize))
                for i, (image, label) in enumerate(zip(images_train[start_idx:end_idx], labels_train[start_idx:end_idx])):
                    output = ut.CNN_forward(image, layers)
                    label_arr = np.zeros(len(output))
                    label_arr[label] = 1
                    batch_predictions.append(output)
                    batch_labels.append(label)
  

                    error =(output- label_arr)
                    cluster_error[i] = error
                avg_Error =  np.mean(cluster_error, axis=0)
                ut.CNN_backward(avg_Error, layers)

            accuracy = ut.calculate_accuracy(batch_predictions, batch_labels)
            accuracies.append(accuracy) 
            print(f"Genauigkeit für Batch {(epoch*num_batches)+batch_idx + 1}: {accuracy:.2f}%\nDurchschnittliche Genauigkeit über alle Batches: { np.mean(accuracies):.2f}%\nHöste Genauigkeit bis Jetzt {accuracies[np.argmax(accuracies)]:.2f}% bei batch {np.argmax(accuracies)+1}")
            

            
            if(accuracy >85 ):
                return (time.time()-st)
            
            learning_rate_idx = int(accuracy // 10)

            lr = lrList[learning_rate_idx]


        
print(train())
test()