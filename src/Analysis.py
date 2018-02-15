import matplotlib.pyplot as plt

import numpy as np
from numpy import std
from scipy.spatial import distance
from sklearn.metrics import f1_score, recall_score,precision_score, roc_curve, auc

class Analysis:
    
    def __init__(self):    
        self.version = 'v1.0'
        
    def f1_score(self,x,y, average = 'weighted'):
        return f1_score(y,x,average=average)
    
    def precision_score(self,x,y, average = 'weighted'):
        return precision_score(y,x, average=average)
    
    def recall_score(self,x,y, average = 'weighted'):
        return recall_score(y,x, average=average)
    
    def rocCurve(self,x,y):
        fpr, tpr, _ = roc_curve(y, x)
        
        return fpr,tpr,auc(fpr, tpr)
    
    def buildPlot(self, data, _params, labels, dependence = None, showAt = 0, showFor = 50, savePath = None,
                  show = True, title = 'Title', xlabel = 'x', ylabel = 'y',grid = True, addPlot = False):
        
        if not addPlot:
            plt.figure()
        
        for i in range(len(data)):
            
            if showFor == 0:
                showFor = len(data[i])
            
            buildFlag = False
            
            if dependence != None:
                if dependence[i]:
                    plt.plot(data[i][0][showAt:showFor],data[i][1][showAt:showFor], _params[i], label = labels[i])
                    buildFlag = True
            
            if not buildFlag:
                plt.plot(data[i][showAt:showFor], _params[i], label = labels[i])
                    
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(grid)

        plt.tight_layout()

        if savePath != None:
            plt.savefig(savePath)

        if show:
            plt.show()
    