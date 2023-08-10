import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from scipy.special import softmax
class ConvolutionLayer:
    def __init__(self,kernel_num,kernel_size) -> None:
        #init Kernels
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(kernel_num,kernel_size,kernel_size) / (kernel_size**2)
        self.biases = np.random.randn(kernel_num)

    def forward(self,images):
        convolution_output  = []
        for image in images:
            for kernel,bias in zip(self.kernels,self.biases):
                kernel_conv = convolve2d(image,kernel,mode='same')
                kernel_conv += bias
                convolution_output.append(kernel_conv) 
            
        return np.array(convolution_output)

class MaxPooling2D:
    def __init__(self,kernel_size) -> None:
        self.kernel_size = kernel_size
    def forward(self,image):
        convolution_output  = []
        pool_size = (1, self.kernel_size, self.kernel_size)
        return maximum_filter(image, size=pool_size, mode='nearest')[::pool_size[0], ::pool_size[1], ::pool_size[2]]
class DenseLayer:
    def __init__(self,output_size,input_size) -> None:
        self.output_size = output_size
        self.input_size = input_size
        self.weights = np.random.randn(self.output_size,self.input_size)
        self.bias = np.random.randn(self.output_size)
    def forward(self,input):
        return np.dot(self.weights,input)+self.bias

class FlattenLayer:
    def forward(input):
        return input.flatten()          

class ReLuLayer:
    def forward(input):
        return np.maximum(0, input)
class softmaxLayer:
    def forward(input):
        return softmax(input)
def CNN_forward(image,layers):
    output = image.reshape(1, image.shape[0], image.shape[1])

    for layer in layers:
        output = layer.forward(output)

    return (output)


def normalizeData(data):    
    data[:, 1:] = data[:, 1:] / 255.0
    return  data 


