import torch
import torch.nn as nn
from torch import nn as nn

from ..cycle_monitoring.cycles import check_cycles

# Models used for the simple MNIST example


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128),
                                nn.PReLU(),
                                nn.Linear(128, 64)
                                )
                                
        

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class SiameseNet(nn.Module):
    def __init__(self, embedding_net, monitor_cycles=False, monitor_cycle_inequalities=False):

        super(SiameseNet, self).__init__()

        self.embedding_net = embedding_net
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(1,2)
        self._softmax = nn.Softmax(dim=1)
        self.bilinear = nn.Bilinear(64,64,10)
        #self.bilinear = nn.Bilinear(32,32,10)
        self.fc3 = nn.Linear(10,2)
        
        self.monitor_cycles = monitor_cycles
        self.monitor_cycle_inequalities = monitor_cycle_inequalities

    def forward(self, x1, x2, cliques_ints=None):

        if self.monitor_cycles or self.monitor_cycle_inequalities:
            assert(cliques_ints),'Clique Index has to be provided if you want to monitor cycles'

        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        
        fc = self.bilinear(output1,output2)
        fc = self.nonlinear(fc)
        fc = self.fc3(fc)
        scores = self._softmax(fc)
        
        # check number of valid/invalid cycles
        cycle_counts_rounded = [] # in case monitor_cycles_rounded = False
        if self.monitor_cycles:
            #scores.to('cpu')
            cycle_counts_rounded = check_cycles(scores, cliques_ints, rounded=True)
            
            '''print('######### Cycle Counts rounded ##########')
            print('Valid Cycles: ', cycle_counts_rounded[0])
            print('Invalid Cycles: ', cycle_counts_rounded[1])'''
        
        cycle_counts = []
        if self.monitor_cycle_inequalities:
            #scores.to('cpu')
            cycle_counts = check_cycles(scores, cliques_ints, rounded=False)
            
            '''print('######### Cycle Counts not rounded ##########')
            print('Valid Cycles: ', cycle_counts[0])
            print('Invalid Cycles: ', cycle_counts[1])'''
        
        return scores, cycle_counts_rounded, cycle_counts, fc


class FullNet(nn.Module):

    def __init__(self,CrfRnn_Multicut,SiameseNet,frontend_device,crf_device,only_frontend=False):
        super(FullNet, self).__init__()

        self.crf_device = crf_device
        self.multicut = CrfRnn_Multicut.to(self.crf_device)
        self.frontend_device = frontend_device
        self.siamese = SiameseNet.to(self.frontend_device)
        self.only_frontend = only_frontend

    def forward(self, x1, x2, edge_index, j_edge_indices, cliques_ints=None):
        #print('Start forward pass')
        #start = timer()
        x1 = x1.to(self.frontend_device)
        x2 = x2.to(self.frontend_device)
        #print('Beginning with Siamese Model')
        siamese_scores, frontend_cycle_counts_rounded, frontend_cycle_counts, fc = self.siamese(x1,x2,cliques_ints)
        #print('Frontend_cycles rounded: ',frontend_cycle_counts_rounded)
        #print('Frontend_cycles: ',frontend_cycle_counts)
        #end = timer()
        #print('Finished Siamese Model, elapsed time: ', end-start)
        if self.only_frontend:
            return siamese_scores, frontend_cycle_counts_rounded, frontend_cycle_counts, None
        else:
            #print('Start Multicut Module')
            #start = timer()
            siamese_scores = siamese_scores.to(self.crf_device)
            fc = fc.to(self.crf_device)
            out, cycle_counts_rounded, cycle_counts = self.multicut(fc, edge_index, j_edge_indices, cliques_ints)
            #end = timer()
            #print('Multicut finished, elapsed time: ', end-start)
            return out, cycle_counts_rounded, cycle_counts, siamese_scores