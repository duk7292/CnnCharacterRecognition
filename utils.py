import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from scipy.special import softmax
class ConvolutionLayer:
    def __init__(self,kernel_num,kernel_size) -> None:
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
    def backward(self,error):
        return error

class MaxPooling2D:
    def __init__(self,kernel_size) -> None:
        self.kernel_size = kernel_size
        self.input = None
    def forward(self,image):
        self.input = image
        convolution_output  = []
        pool_size = (1, self.kernel_size, self.kernel_size)
        return maximum_filter(image, size=pool_size, mode='nearest')[::pool_size[0], ::pool_size[1], ::pool_size[2]]
    

    def backward(self,error):
        dinput = np.zeros(self.input.shape)
        for img in range(error.shape[0]):
            pass
        return error
class DenseLayer:
    def __init__(self,output_size,input_size):
        self.output_size = output_size
        self.input_size = input_size

        self.weights = np.random.randn(self.output_size,self.input_size) / np.sqrt(self.input_size)
        self.bias = np.random.randn(self.output_size,1)

        self.input = None
    def forward(self,input):
        
        self.input = input
       
        
        return np.dot(self.weights,input)+(self.bias.reshape(self.output_size))

    def backward(self,error):
        error = error.reshape(-1, 1)
        input_reshaped = self.input.reshape(1, -1)
        self.dW = np.dot(error, input_reshaped)
        self.db = np.sum(error, axis=1)
        dA_prev = np.dot(self.weights.T, error)
        return dA_prev.ravel() 

class FlattenLayer:
    def __init__(self) -> None:
        self.input = None
    def forward(self,input):
        self.input = input
        return input.flatten()
     
    def backward(self,error):
        return error.reshape(self.input.shape)       

class ReLuLayer:
    def __init__(self) -> None:
        self.input = None
    def forward(self,input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self,error):
        error[self.input <= 0] = 0
       
        return error
    
class softmaxLayer:
    def __init__(self) -> None:
        self.input = None
        self.softmax_output = None
    def forward(self,input):
        self.softmax_output = softmax(input)
        self.input = input
        return self.softmax_output
    
    def backward(self,true_labels):
        gradient = self.softmax_output - true_labels
        return gradient
def CNN_forward(image,layers):
    output = image.reshape(1, image.shape[0], image.shape[1])

    for layer in layers:
        
        output = layer.forward(output)

    return (output)
def CNN_backward(error_output,layers):
    error =error_output
    for layer in reversed(layers):
        
        error = layer.backward(error)
        print(error.shape,layer)

def normalizeData(data):    
    data[:, 1:] = data[:, 1:] / 255.0
    return  data 


