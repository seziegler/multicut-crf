import torch.nn as nn
import itertools
import numpy as np

from ..utils.rcf_crf import intervening_contour_cue, intervening_contour_cue_fast, mask_nms
from ..utils.crf import get_graph_idx
from ..cycle_monitoring.cycles import check_cycles

## Model used for Edge Detection and Segmentation

class RCF_CRFasRNN(nn.Module):

    def __init__(self, rcf, crf, only_rcf=False, only_edgemap=False, rcf_backbone='Resnet', k = 8):

        super(RCF_CRFasRNN, self).__init__()

        self.rcf = rcf
        self.crf = crf
        self.only_rcf = only_rcf
        self.only_edgemap = only_edgemap
        self.rcf_backbone = rcf_backbone
        self.k = k
        
      

    def forward(self, img, edge_gt=None, shape_dict={}, contour_cue_dict={}, nms=None):
    
        k = self.k

        #### RCF ####
        #print('RCF starts')

        h, w = nms.size()[1:]
        
        if self.rcf_backbone=='Resnet':
            rcf_outputs = self.rcf(img, (h, w)) # returns all side outputs and fuse
            
        elif self.rcf_backbone=='VGG':
            rcf_outputs = self.rcf(img) # returns all side outputs and fuse

        if self.only_edgemap:
            # regular RCF
            return rcf_outputs
        
        # only use fuse as final edgemap
        edgemap = rcf_outputs[-1]
        

        #print('RCF done. NMS starts')

        # Non Max Suppression
        nms = nms.to(edgemap.get_device() if edgemap.get_device()!=-1 else 'cpu')
        edgemap_thin = mask_nms(edgemap, nms[0])

        #print('NMS done. Contour cue starts.')

        img_shape = edgemap_thin.size()

        #print('Shape: ',img_shape)
        
        if img_shape in contour_cue_dict:
            #print('Using fast version')
            edges, unaries, edges_c, unaries_c, cliques, seg_labels, contour_cue_dict = intervening_contour_cue_fast(edgemap_thin, edge_gt, contour_cue_dict)
        else:
            #print('Using slow version')
            edges, unaries, edges_c, unaries_c, cliques, seg_labels, contour_cue_dict = intervening_contour_cue(edgemap_thin, edge_gt, k=k, contour_cue_dict=contour_cue_dict)
        
        #print('Contour Cue done.')

        if self.only_rcf:
            #img_shape = edgemap.size()
            # check invalid cycles
            # get clique_index 
            if img_shape in shape_dict:
                _, _, cliques_ints = shape_dict[img_shape]
            else:
                int2edge_mapping = dict(zip(edges_c,np.arange(0,len(edges_c),1)))
                cliques_ints = [list(map(int2edge_mapping.get,list(itertools.combinations(c,2)))) for c in cliques]
                shape_dict[img_shape] = None,None,cliques_ints

            cycle_counts_rounded = check_cycles(unaries_c,cliques_ints,rounded=True)
            cycle_counts = check_cycles(unaries_c,cliques_ints,rounded=False)

            return rcf_outputs, edges, unaries, edges_c, unaries_c, cliques, seg_labels, cycle_counts_rounded, cycle_counts, shape_dict, contour_cue_dict

        #### CRF ####
        # prepare graph indices
        # save indices in shape_dict so that calculation is only done once for each shape

        #print('Start CRF.')

        #print(img_shape)
        #if img_shape in shape_dict and self.training:
        if img_shape in shape_dict:
            #print('Shape was in dict')
            edge_index, j_edge_indices, cliques_ints = shape_dict[img_shape]
            
        else:
            #print('Shape was not in dict')
            edge_index, j_edge_indices, cliques_ints = get_graph_idx(edges_c, cliques)
            if self.training:
                shape_dict[img_shape] = edge_index, j_edge_indices, cliques_ints

        updated_unaries_c, cycle_counts_rounded, cycle_counts = self.crf(unaries_c,edge_index,j_edge_indices,cliques_ints)
        #print('CRF done')
        return rcf_outputs, edges, unaries, edges_c, updated_unaries_c, cliques, seg_labels, cycle_counts_rounded, cycle_counts, shape_dict, contour_cue_dict