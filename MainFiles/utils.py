import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import cupy as cp
import time
class ConvolutionLayer:
    def __init__(self,kernel_num,kernel_size,learning_rate) -> None:
        self.learning_rate = learning_rate
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) * np.sqrt(2. / (kernel_size * kernel_size))


        self.biases = np.zeros(kernel_num) 


        self.input = None
    def forward(self,images):
        self.input = images
        convolution_output  = []
        
        for kernel,bias in zip(self.kernels,self.biases):
            for image in images:
                kernel_conv =  convolve2d(image, kernel, mode='same')#conv2d_same(image,kernel)

                
                kernel_conv += bias

                convolution_output.append(kernel_conv) 
        np_convolution_output = np.array(convolution_output)
        
        return np_convolution_output
    def backward(self, errors):
        dKernels = np.zeros_like(self.kernels)
        dBiases = np.zeros_like(self.biases)
        dInputs = np.zeros_like(self.input)
        
        
        for i, (kernel, bias) in enumerate(zip(self.kernels, self.biases)):
            
            for j, image in enumerate(self.input):
                dKernels[i] +=convolve2d(image,  errors[i], mode='valid') #conv2d_valid(image, errors[i])
                dBiases[i] += np.sum(errors[i])

                
                

                rotated_kernel = np.rot90(kernel, 2)
                dInputs[j] += convolve2d(errors[i], rotated_kernel, mode='same')#conv2d_same(errors[i], rotated_kernel)
        self.kernels -= self.learning_rate * dKernels
        self.biases -= self.learning_rate * dBiases
        return dInputs

class MaxPooling2D:
    def __init__(self,kernel_size) -> None:
        self.kernel_size = kernel_size
        self.input = None
        self.max_mask = None
    def forward(self,images):
        self.input = images
        pooled_output = np.zeros((len(images), *maxpooling2d(images[0], 2).shape))
        for idx, image in enumerate(images):
            pooled_output[idx] = maxpooling2d(image, 2)
        
        

        enlarged_pooled_output = pooled_output.repeat(1, axis=0).repeat(self.kernel_size, axis=1).repeat(self.kernel_size, axis=2)
        self.max_mask = (images == enlarged_pooled_output).astype(int)
        correct_max_mask_sum = int((images.shape[0]*(images[0].shape[0]*images[0].shape[1]))/self.kernel_size**2)
        if(np.sum(self.max_mask) > correct_max_mask_sum):
            self.max_mask =  correct_max_mask(self.max_mask,self.kernel_size)
        
        return pooled_output

    def backward(self,error):
        dinput = np.zeros(self.input.shape)
        for n in range(error.shape[0]):
            for i in range(error.shape[1]):
                for j in range(error.shape[2]):
                    dinput[n, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size] = error[n, i, j] * self.max_mask[n, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size]        
        return dinput
class DenseLayer:
    def __init__(self,output_size,input_size,learning_rate,cupy_enabled = False):
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.input_size = input_size
        self.cupy_enabled = cupy_enabled
        if(cupy_enabled):
            self.weights = cp.random.randn(self.output_size, self.input_size) * cp.sqrt(2. / self.input_size)
            self.bias = cp.zeros(self.output_size)
        else:
            self.weights = np.random.randn(self.output_size, self.input_size) * np.sqrt(2. / self.input_size)
            self.bias = np.zeros(self.output_size) 


    
        self.input = None
    def forward(self, input):
        self.input = np.array(input)  # Eingang als NumPy-Array
        if(self.cupy_enabled):
            
            input_cp = cp.asarray(self.input)
            

            # Berechnung mit CuPy
            output_cp = cp.dot(self.weights, input_cp) + (self.bias.reshape(self.output_size))

            # Konvertiere das CuPy-Array zurück zu einem NumPy-Array
            output = cp.asnumpy(output_cp)

        else:  
        # Berechnung mit NumPy
            output = np.dot(self.weights, self.input) + (self.bias.reshape(self.output_size))

        return output

    def backward(self,error):
        error = error.reshape(-1, 1)
        input_reshaped = self.input.reshape(1, -1)

        if(self.cupy_enabled):
            error_cp = cp.asarray(error)
            input_reshaped_cp = cp.asarray(input_reshaped)
            
            # Berechnung mit CuPy
            dW = cp.dot(error_cp, input_reshaped_cp)
            db = cp.sum(error_cp, axis=1)
            dA_prev_cp = cp.dot(cp.transpose(self.weights), error_cp)

            # Konvertiere die CuPy-Arrays zurück zu NumPy-Arrays
            
            dA_prev = cp.asnumpy(dA_prev_cp)


        else:
            dW = np.dot(error, input_reshaped)
            db = np.sum(error, axis=1)
            dA_prev = np.dot(self.weights.T, error)
        
        self.bias -= self.learning_rate * db
        self.weights -= self.learning_rate *  dW
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
    
    def backward(self, error):
        
        error[self.input < 0] = 0 
        return  error



  


class SoftmaxLayer:
    def __init__(self, axis=-1) -> None:
        self.output = None
        self.axis = axis

    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=self.axis, keepdims=True))  # Numerische Stabilität
        self.output = exps / np.sum(exps, axis=self.axis, keepdims=True)
        return self.output

    def backward(self, error):
        grad = self.output * (error - np.sum(error * self.output, axis=self.axis, keepdims=True))
        return grad


def CNN_forward(image,layers):
    output = image.reshape(1, image.shape[0], image.shape[1])

    for layer in layers:     
        output = layer.forward(output)
 
    return (output)
def CNN_backward(error_output,layers):
    error =error_output
    #print("mean:",np.mean(error),"of start Error")
    #print("var:",np.var(error),"of start Error")
    #print("_____________________")
    for layer in reversed(layers):
        error = layer.backward(error) 
        #print("mean:",np.mean(error),"of",type(layer))
        #print("var:",np.var(error),"of",type(layer))
        #print("sum:",np.sum(np.abs(error)),"of",type(layer))

def normalizeData(data):    
    data[:, 1:] = data[:, 1:] / 255.0
    return  data 


def calculate_accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_labels == labels)
    accuracy = (correct_predictions / len(labels)) * 100
    return accuracy

def int_to_letter(n):
    if 0 <= n <= 25:  
        return chr(97 + n)
    else:
        raise ValueError("Die Zahl sollte zwischen 0 und 25 liegen.")
def conv2d_valid(input_array, kernel, output_size=(3, 3)):
     

    x, y = input_array.shape
    a, b = kernel.shape

    output_x, output_y = output_size

    output = np.zeros((output_x, output_y))

    for i in range(output_x):
        for j in range(output_y):
            if i + a <= x and j + b <= y:
                output[i, j] = np.sum(input_array[i:i+a, j:j+b] * kernel)
            else:
                # Bereich außerhalb der Eingabematrix mit Nullen auffüllen
                temp_array = np.zeros((a, b))
                temp_array[:min(x-i, a), :min(y-j, b)] = input_array[i:min(i+a, x), j:min(j+b, y)]
                output[i, j] = np.sum(temp_array * kernel)

    return output



def conv2d_same(input_array, kernel):
    
    x, y = input_array.shape
    a, b = kernel.shape

    pad_x = a // 2
    pad_y = b // 2

    padded_input = np.pad(input_array, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')

    output = np.zeros_like(input_array)

    for i in range(x):
        for j in range(y):
            output[i, j] = np.sum(padded_input[i:i+a, j:j+b] * kernel)

    return output

def maxpooling2d(input_matrix, kernel_size):
    # Get the dimensions of the input matrix
    rows, cols = input_matrix.shape
    
    # Calculate the dimensions of the pooled matrix
    pooled_rows = rows // kernel_size
    pooled_cols = cols // kernel_size
    
    # Initialize the output matrix with zeros
    output_matrix = np.zeros((pooled_rows, pooled_cols))
    
    # Perform max pooling and place the result in the center of the output matrix
    for i in range(pooled_rows):
        for j in range(pooled_cols):
            output_matrix[i , j ] = np.max(input_matrix[(kernel_size*i):(kernel_size*i)+kernel_size, (kernel_size*j):(kernel_size*j)+kernel_size])
    
    return output_matrix

def correct_max_mask(max_masks, Kernel_size):
    correct_max_mask = np.zeros_like(max_masks)
    for i, max_mask in enumerate(max_masks):
        for y in range(max_mask.shape[0]):
            for x in range(max_mask.shape[1]):
                s_x = x // Kernel_size
                s_y = y // Kernel_size
                patch = correct_max_mask[i, s_y*Kernel_size:s_y*Kernel_size+Kernel_size, s_x*Kernel_size:s_x*Kernel_size+Kernel_size]
                patch_sum = np.sum(patch)
                if patch_sum == 0:
                    correct_max_mask[i, y, x] = max_mask[y, x]
    return correct_max_mask
