import numpy as np
import progressbar
import copy, time
#import numba

class DataHandler:

    def __init__(self):
        self.version = '0.1'
        
    def maxNorm(self,data, _max = None):
        
        if _max == None:
            _max = max([max(d) for d in data])

        result = copy.deepcopy(data)
        
        print('Normalization...')
        
        bar = progressbar.ProgressBar(max_value = int(len(result)))
        time.sleep(0.5)
        for i in range(len(result)):
            result[i] = np.array(result[i]) / _max
            
            bar.update(i)
            
        bar.finish()    
        return result
                
    def split(self,data,rate = 0.7):
        
        sep = int(len(data) * rate)
        
        return data[:sep], data[sep:]
    
    #@numba.jit
    def toYCbCr(self,image):
        
        r = np.array(image[:int(len(image) / 3)])
        g = np.array(image[int(len(image) / 3):int(len(image) / 3) * 2])
        b = np.array(image[int(len(image) / 3) * 2:])
        
        y = np.zeros(int(len(image) / 3))
        
        for pixel in range(len(r)):
            temp = int(0.299 * r[pixel] + 0.587 * g[pixel] + 0.114 * b[pixel])
            
            if temp < 0: temp = 0
            if temp > 255 : temp = 255
            
            y[pixel] = temp
            
            
        return y