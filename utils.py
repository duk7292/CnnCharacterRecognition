import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

class ConvolutionLayer:
    def __init__(self,kernel_num,kernel_size,learning_rate) -> None:
        self.learning_rate = learning_rate
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(kernel_num,kernel_size,kernel_size) / (kernel_size**2)
        self.biases = np.random.randn(kernel_num)
        self.input = None
    def forward(self,images):
        self.input = images
        convolution_output  = []
        
        for kernel,bias in zip(self.kernels,self.biases):
            for image in images:
                kernel_conv = convolve2d(image,kernel,mode='same')
                kernel_conv += bias
                convolution_output.append(kernel_conv) 
        
        return np.array(convolution_output)
    def backward(self,error):
        num_images = error.shape[0] // self.kernel_num
    
        error_reshaped = error.reshape(self.kernel_num, num_images, error.shape[1], error.shape[2])
      
       
        dbiases = np.sum(error_reshaped, axis=(1,2,3))
        dKernels_accumulator = np.zeros_like(self.kernels)

        errors_per_kernel = error.shape[0] // self.kernel_num
        for kernel_idx ,kernel in enumerate(self.kernels):
            dkernel_sum = np.zeros(kernel.shape)
            for img_idx,image in enumerate(self.input):
                
                dKernel_full = convolve2d(image, error[kernel_idx*errors_per_kernel+img_idx], mode='full')

                kernel_center_size = self.kernel_size

                start_idx = (dKernel_full.shape[0] - kernel_center_size) // 2
                end_idx = start_idx + kernel_center_size

             
                dKernel = dKernel_full[start_idx:end_idx, start_idx:end_idx]
               
                dkernel_sum += dKernel
            dKernels_accumulator[kernel_idx] = dkernel_sum / errors_per_kernel
        
        self.kernels -= self.learning_rate * dKernels_accumulator
        self.biases -= self.learning_rate * dbiases
        

        dInput = np.zeros(self.input.shape)

        for kernel_idx, kernel in enumerate(self.kernels):
            for img_idx, image in enumerate(self.input):
                dInput[img_idx] += convolve2d(error[kernel_idx*errors_per_kernel+img_idx], np.rot90(kernel, 2), mode='same')
        return dInput,dKernel,dbiases

class MaxPooling2D:
    def __init__(self,kernel_size) -> None:
        self.kernel_size = kernel_size
        self.input = None
        self.max_mask = None
    def forward(self,images):
        self.input = images
        convolution_output  = []
        pool_size = (1, self.kernel_size, self.kernel_size)

        pooled_output = maximum_filter(images, size=pool_size, mode='nearest')[::pool_size[0], ::pool_size[1], ::pool_size[2]]
        
        
        enlarged_pooled_output = pooled_output.repeat(1, axis=0).repeat(self.kernel_size, axis=1).repeat(self.kernel_size, axis=2)
        self.max_mask = (images == enlarged_pooled_output).astype(int)

        return pooled_output

    def backward(self,error):
        dinput = np.zeros(self.input.shape)
        for n in range(error.shape[0]):
            for i in range(error.shape[1]):
                for j in range(error.shape[2]):
                    dinput[n, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size] = error[n, i, j] * self.max_mask[n, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size]        
        return dinput,0,0
class DenseLayer:
    def __init__(self,output_size,input_size,learning_rate):
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.input_size = input_size

        self.weights = np.random.randn(self.output_size,self.input_size) / np.sqrt(self.input_size)
        self.bias = np.random.randn(self.output_size)

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
        
        self.bias -= self.learning_rate * self.db
        self.weights -= self.learning_rate *  self.dW
        return dA_prev.ravel(),self.dW,self.db

class FlattenLayer:
    def __init__(self) -> None:
        self.input = None
    def forward(self,input):
        self.input = input
        return input.flatten()
     
    def backward(self,error):
        return error.reshape(self.input.shape),0,0

class ReLuLayer:
    def __init__(self) -> None:
        self.input = None
    def forward(self,input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self,error):
        error[self.input <= 0] = 0
       
        return error,0,0
    
class softmaxLayer:
    def __init__(self) -> None:
        self.input = None
        self.softmax_output = None
    def softmax(self,x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        e_x = np.exp(x - np.max(x))  
        return e_x / e_x.sum(axis=1, keepdims=True)
    def forward(self,input):
        self.input = input
        self.softmax_output = self.softmax(input)
        
        return self.softmax_output
    
    def backward(self,true_labels):
        gradient = self.softmax_output - true_labels
        return gradient,0,0
def CNN_forward(image,layers):
    output = image.reshape(1, image.shape[0], image.shape[1])

    for layer in layers:
        
        output = layer.forward(output)

    return (output)
def CNN_backward(error_output,layers):
    error =error_output
    for layer in reversed(layers):
        error = layer.backward(error)[0]

def normalizeData(data):    
    data[:, 1:] = data[:, 1:] / 255.0
    return  data 


def calculate_accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_labels == labels)
    accuracy = (correct_predictions / len(labels)) * 100
    return accuracy