import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import utils as ut
import matplotlib.pyplot as plt
learning_rate = 0.000005
def main():
    # CSV-Datei einlesen
    dataRaw = pd.read_csv('A_Z_Handwritten_Data.csv').to_numpy().astype(float)
    np.random.shuffle(dataRaw)
    fraction = 0.5
    dataRaw = dataRaw[:int(len(dataRaw) * fraction)]
    data_Normalized = ut.normalizeData(dataRaw)
    data_Normalized = np.array(data_Normalized)

    labels = data_Normalized[:, 0].astype(int)
    images_1d = data_Normalized[:, 1:]
    
    image_size = 28
    images_2d = images_1d.reshape(-1, image_size, image_size)

    labels_train, labels_test, images_train, images_test = train_test_split(labels, images_2d, test_size=0.2)
 

    

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
    def train_in_batches(images_train, labels_train, layers, batch_size=300):
        global learning_rate
       
        num_batches = len(images_train) // batch_size
        accuracies = []

        for batch_idx in range(num_batches):
            
            batch_predictions = []
            batch_labels = []
         
               

         
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            last_label = 0
            for i, (image, label) in enumerate(zip(images_train[start_idx:end_idx], labels_train[start_idx:end_idx])):
                
                output = ut.CNN_forward(image, layers)[0]
                
                label_arr = np.zeros(len(output))
                label_arr[label] = 1
                
                
                batch_predictions.append(output)
                batch_labels.append(last_label)

                error =  label_arr  -output
                ut.CNN_backward(error, layers)
                last_label = label
            for i in range(len(batch_labels)):

                print(batch_labels[i], np.argmax(batch_predictions[i]))
           
            batch_predictions = np.array(batch_predictions)
            batch_labels = np.array(batch_labels)

            print((batch_predictions[0]),batch_labels[0],np.argmax(batch_predictions[0]))

        
            accuracy = ut.calculate_accuracy(batch_predictions, batch_labels)
            accuracies.append(accuracy)
            print(f"Genauigkeit für Batch {batch_idx + 1}: {accuracy:.2f}%")

        return np.mean(accuracies)

    average_accuracy = train_in_batches(images_train, labels_train, layers)
    print(f"Durchschnittliche Genauigkeit über alle Batches: {average_accuracy:.2f}%")
        
        
        

if __name__ == "__main__":
    main()