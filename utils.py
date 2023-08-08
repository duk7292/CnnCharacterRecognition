import numpy as np
from scipy.signal import convolve2d

class ConvolutionLayer:
    def __init__(self,kernel_num,kernel_size) -> None:
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        kernels = np.random.randn(kernel_num,kernel_size,kernel_size) / (kernel_size**2)

        print(kernels[0])






def normalizeData(data):
    
    data[:, 1:] = data[:, 1:] / 255.0
    return  data 


