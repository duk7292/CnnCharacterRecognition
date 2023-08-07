import numpy as np

def normalizeData(data):
    
    data[:, 1:] = data[:, 1:] / 255.0
    return  data 


