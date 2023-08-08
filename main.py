import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import utils as ut
import matplotlib.pyplot as plt
def main():
    # CSV-Datei einlesen
    dataRaw = pd.read_csv('A_Z_Handwritten_Data.csv').to_numpy().astype(float)
    np.random.shuffle(dataRaw)
    fraction = 0.1
    dataRaw = dataRaw[:int(len(dataRaw) * fraction)]
    data_Normalized = ut.normalizeData(dataRaw)
    data_Normalized = np.array(data_Normalized)

    labels = data_Normalized[:, 0].astype(int)
    images_1d = data_Normalized[:, 1:]

    image_size = 28
    images_2d = images_1d.reshape(-1, image_size, image_size)

    labels_train, labels_test, images_train, images_test = train_test_split(labels, images_2d, test_size=0.2)
 
    layers = [
        ut.ConvolutionLayer(32,3),
        ut.ReLuLayer,
        ut.MaxPooling2D(2),
        ut.ConvolutionLayer(32,3),
        ut.ReLuLayer,
        ut.MaxPooling2D(2),
        ut.FlattenLayer
            ]

    for i, (image, label) in enumerate(zip(images_train, labels_train)): 
        output = ut.CNN_forward(image,layers)
        print(output.shape)
        
            
    

if __name__ == "__main__":
    main()