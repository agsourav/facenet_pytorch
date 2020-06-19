import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):        #classification margin
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average = True): 
        try:    
            distance_positive = (anchor - positive).pow(2).sum(1)
            distance_negative = (anchor - negative).pow(2).sum(1)
            losses = F.relu(distance_positive - distance_negative + self.margin)
            return losses.mean() if size_average else losses.sum()       
        except:
            print('Error computing loss')    
            exit()


