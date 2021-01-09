import pickle
import torch
import numpy as np
import os
import argparse

from src.utils.rcf_crf import intervening_contour_cue
from src.utils.crf import get_graph_idx

if __name__ == '__main__':

    ### Args ###
    
    parser = argparse.ArgumentParser(description='Define for which shape and k you want to save the dict.')
    parser.add_argument('--k', default=2, help='Distance to most distant pixel from source pixel', type=int)
    parser.add_argument('--dev', default='cpu', help='Device type (cuda or cpu)', type=str)
    parser.add_argument('--prefix', default='', help='if you want an extra prefix for naming, e.g. "isbi" will create isbi_contour_cue_dict_k.p, default leaves this empty', type=str)
    
    args = parser.parse_args()

    device=args.dev
    
    # BSDS500
    #size = (321, 481)
    size = (481, 321)
    
    # ISBI2012
    #size = (512,512)

    k = args.k
    prefix = args.prefix
    
    edge_map = torch.rand([1,1,size[0],size[1]]).to(device)
    edge_gt = torch.rand([1,1,size[0],size[1]]).to(device)
    img_shape = edge_map.size()
    
    # if dict name exists -> load, else: dict = {}, check dict name with default + _k
    # name the dicts
    if len(prefix)==0:
        ccd_name = 'dicts/contour_cue_dict_'+str(k)+'.p'
        sd_name = 'dicts/shape_dict_'+str(k)+'.p'
    else:
        ccd_name = 'dicts/'+str(prefix)+'_contour_cue_dict_'+str(k)+'.p'
        sd_name = 'dicts/'+str(prefix)+'_shape_dict_'+str(k)+'.p'
    # check existence, init new or load existing one for update
    if os.path.isfile(ccd_name):
        contour_cue_dict = pickle.load(( open( ccd_name, "rb" ) ))
    else:
        contour_cue_dict = {}
    if os.path.isfile(sd_name):
        shape_dict = pickle.load(( open( sd_name, "rb" ) ))
    else:
        shape_dict = {}
    
    
    if img_shape not in contour_cue_dict:
        _, _, edges_c, _, cliques, _, contour_cue_dict = intervening_contour_cue(edge_map, edge_gt, k, prior=0.45, contour_cue_dict=contour_cue_dict, dict_update=True)
        # save contour_cue_dict
        pickle.dump( contour_cue_dict, open( ccd_name, "wb" ) )
    else:
        print('Contour Cue Dict: The specified img_shape for the specified k is already in the dictionary')
    
    
    if img_shape not in shape_dict:
        edge_index, j_edge_indices, cliques_ints = get_graph_idx(edges_c, cliques)
        shape_dict[img_shape] = edge_index, j_edge_indices, cliques_ints   
        # save shape_dict
        pickle.dump( shape_dict, open( sd_name, "wb" ) )
    else:
        print('CRF Coords Dict: The specified img_shape for the specified k is already in the dictionary')
    
    print('Done.')