import torch
import torchvision
import argparse

from src.models.rcf_crf import RCF_CRFasRNN
from src.models.crf_multicut import CrfMulticut
from rcf_vgg.models import RCF as rcf_vgg
from rcf_resnet import models
from rcf_resnet.data_loader import BSDS_RCFLoader as resnet_data_loader
from rcf_vgg.data_loader import BSDS_RCFLoader as vgg_data_loader
from torch.utils.data import DataLoader
from os.path import split

import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.utils.rcf_crf import combine_edges, create_lifted_mc_problem
from scipy.io import savemat
import subprocess
import h5py
import os
from collections import defaultdict
from itertools import compress
import cv2


def load_model(path_to_model='', only_frontend=False, no_crf=False, backbone='Resnet'):

    """
    Loads specified model.
    Args:
         path_to_model:     Path to the trained pytorch model (not needed if you only want to use the pretrained frontend model)
         only_frontend:     True if your model is only the frontend model (original model, no Intervening Contour Cue and CRF)
         no_crf:            True if your model was only trained with RCF and Intervening Contour Cue (no CRF)
         backbone:          The backbone used by the RCF (Resnet or VGG)

    Returns:
         model
    """

    assert backbone in ['Resnet','VGG']
      
    # load frontend model 
    if backbone=='Resnet':
        
        # load model with Resnet101 backbone  
        rcf = models.resnet101(pretrained=False).cuda()
        
        # load pretrained version of frontend if you want to use only the frontend model
        if only_frontend:
            path = 'rcf_resnet/only-final-lr-0.01-iter-130000.pth'
            checkpoint = torch.load(path)
            rcf.load_state_dict(checkpoint)
        
    elif backbone=='VGG':
        
        # load model trained with VGG16 backbone  
        rcf = rcf_vgg().cuda()
        
        # load pretrained version of frontend if you want to use only the frontend model
        if only_frontend:          
            #path = 'pretrained/RCF_pretrained/pretrained_RCFcheckpoint_epoch12.pth'
            path = 'rcf_vgg/tmp/RCF_stepsize3/checkpoint_epoch8.pth'
            #path = 'rcf_vgg/tmp/RCF/checkpoint_epoch8.pth'
            checkpoint = torch.load(path)
            rcf.load_state_dict(checkpoint['state_dict'])
    
    # load full model
    if not only_frontend:
        
        assert len(path_to_model)!=0, 'You need to specify the path to your trained model or set only_frontend to True'
        # load pretrained model
        pretrained_dict = torch.load(path_to_model)#,map_location=torch.device('cpu'))
        p = pretrained_dict['p']
            
        model = RCF_CRFasRNN(rcf, CrfMulticut(5, True, True, p), only_rcf=no_crf, only_edgemap=False, rcf_backbone=backbone).cuda()
        model.load_state_dict(pretrained_dict['state_dict'])
    
    else:
        # model that only uses frontend
        model = RCF_CRFasRNN(rcf, CrfMulticut(5, True, True), only_rcf=True, only_edgemap=False, rcf_backbone=backbone).cuda()
    
    return model


def load_bsds_testset(loader):
    
    
    test_dataset = loader(split="test")
    

    test_loader = DataLoader(
            test_dataset, batch_size=1,
            num_workers=8, drop_last=True,shuffle=False)

    with open('data/bsds/test_list_only_img.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    return test_loader
    
def load_bsds_trainset(loader):
    '''
    in case you want to predict on the trainset
    '''
    
    train_dataset = loader(split="train")
    

    train_loader = DataLoader(
            train_dataset, batch_size=1,
            num_workers=8, drop_last=True,shuffle=False)

    return train_loader

def load_dicts():
        
    #shape_dict = {}
    #contour_cue_dict = {}
    #shape_dict = pickle.load( open( "dicts/shape_dict_8.p", "rb" ) )
    #contour_cue_dict = pickle.load( open( "dicts/contour_cue_dict_8.p", "rb" ) )
    shape_dict = pickle.load( open( "dicts/full_line_shape_dict_8.p", "rb" ) )
    contour_cue_dict = pickle.load( open( "dicts/full_line_contour_cue_dict_8.p", "rb" ) )
    

    return shape_dict, contour_cue_dict
    

def predict_edgemaps(model, test_loader, edgemap_path, mat_path):

    model.only_edgemap = True
    
    # uncomment for training data
    #for b, (img, _, img_name, nms) in enumerate(test_loader):
    for b, (img, img_name, nms) in enumerate(test_loader):

        img = img.to('cuda')

        with torch.no_grad():
            model.eval()

            # predict
            res = model(img,nms=nms)
            fuse = res[-1]

            # save predicted maps
            torchvision.utils.save_image(fuse[0, :, :, :], edgemap_path+'/{}.png'.format(img_name[0].split('/')[-1]))
            #torchvision.utils.save_image(fuse[0, :, :, :], edgemap_path+'/{}.png'.format(img_name[0]))
            
            # mat file
            fuse = fuse.squeeze().detach().cpu().numpy()
            savemat(os.path.join(mat_path, '{}.mat'.format(img_name[0].split('/')[-1])), {'result': fuse})
            

def predict_edgemaps_multiscale(model, test_loader, edgemap_path, mat_path):

    model.only_edgemap = True

    for b, (img, img_name, nms) in enumerate(test_loader):

        #img = img.to('cuda')
        
        scale = [0.5, 1.0, 1.5]
        multi_fuse = np.zeros(nms[0].shape, np.float32)

        with torch.no_grad():
            model.eval()
            
            for k in range(0, len(scale)):
            
                im_ = cv2.resize(img[0].numpy().transpose((2,1,0)), None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2,1,0))
                #print(im_.shape)
                im_ = torch.tensor(im_, device='cuda')
                #print(im_.size())
                
                
                nms = nms.float()
                
                nms_ = cv2.resize(nms.numpy().transpose((2,1,0)), None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                #print(nms_.shape)
                nms_ = nms_.transpose((1,0))
                #print(nms_.shape)

                # predict
                res = model(im_.unsqueeze(0),nms=torch.tensor(nms_).unsqueeze(0))
                fuse = res[-1]
                #print(fuse.size())
                #fuse = cv2.resize(fuse[0,0].cpu().numpy(), (img.shape[3], img.shape[2]), interpolation=cv2.INTER_LINEAR)
                fuse = cv2.resize(fuse[0,0].cpu().numpy(), (nms.shape[2], nms.shape[1]), interpolation=cv2.INTER_LINEAR)
                #print(multi_fuse.shape)
                #print(fuse.shape)
                multi_fuse += fuse
                #print(torch.tensor(multi_fuse).size())
            
            multi_fuse /= len(scale)
            #multi_fuse = 255 * (multi_fuse) 
            multi_fuse = torch.tensor(multi_fuse).unsqueeze(0).unsqueeze(0)

            # save predicted maps
            torchvision.utils.save_image(multi_fuse[0, :, :, :], edgemap_path+'/{}.png'.format(img_name[0].split('/')[-1]))
            
            # mat file
            multi_fuse = multi_fuse.squeeze().detach().cpu().numpy()
            savemat(os.path.join(mat_path, '{}.mat'.format(img_name[0].split('/')[-1])), {'result': multi_fuse})

            
def make_segmentation(model, test_loader, shape_dict, contour_cue_dict, path_mc_problem, path_seg):
    
    '''
    GAEC+KLj segmentation
    '''

    model.only_edgemap = False
    

    for b, (img, img_name, nms) in enumerate(test_loader):
        

        # transformed img
        img = img.to('cuda')

        #orig_img = Image.open('./data/bsds/{}.jpg'.format(img_name[0]))

        with torch.no_grad():
            model.eval()

            _,_,edges_only_neighbors,_,_,_ = contour_cue_dict[nms.unsqueeze(0).size()]
            #_,_,edges_only_neighbors,_,_,_ = contour_cue_dict[img[:,0].unsqueeze(0).size()]
            # direct neighbors
            edges_n = list(edges_only_neighbors.keys())
            unaries_n = list(edges_only_neighbors.values())
            unaries_n = torch.stack(unaries_n)
            unaries_n = torch.stack((1-unaries_n,unaries_n),dim=1)
            unaries_n = unaries_n.to('cuda')
            nb_neighbors_edges = len(edges_n)
            
            if len(nms.size())==4:
                nms = nms.squeeze().unsqueeze(0)

            height = nms.size()[1]
            width = nms.size()[2]

            # predict
            _, edges, unaries, edges_c, updated_unaries_c, _, _, _, _, _, _ = model(img,None,shape_dict,contour_cue_dict,nms)
            combined_edges, combined_unaries = combine_edges([edges_n,edges,edges_c],[unaries_n,unaries,updated_unaries_c])

            # create multicut problem txt input files
            filename = create_lifted_mc_problem((height,width), combined_edges, combined_unaries, nb_neighbors_edges, img_name, path_mc_problem)
            #print(filename,'created.')

            # solve multicut
            filename_out = path_mc_problem+'/mc_result_{}.h5'.format(img_name[0].split('/')[-1])         
            subprocess.call(['./graph_lib/graph/build/solve-regular-lifted', '-i', filename, '-o', filename_out])
      

            # load result
            f = h5py.File(filename_out, 'r')
            labels = np.array(f['labels'])
            labels = labels.reshape(height,width)
            try:
                # will fail if all labels are zero
                energy = np.array(f['energy-value'])
            except:
                energy = 0.0


            # save labels as .mat for evaluation in matlab
            mat_labels = {'seg': labels, 'energy': energy}
            savemat(path_seg+'/{}.mat'.format(img_name[0].split('/')[-1]), mat_labels)      

            # clean up mc_problem files (ca. 85 MB per file)
            os.remove(filename)
            os.remove(filename_out)

            print('Image {} processed. Result successfully saved.'.format(img_name[0]))
            

def CCL_segmentation(model, test_loader, shape_dict, contour_cue_dict, threshold, save_path):

    if threshold=='multiple':
        t_list = [0.050,0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500,
                  0.520,0.540,0.560,0.580,0.600,0.620,0.640,0.660,0.680,0.700,
                  0.720,0.740,0.760,0.780,0.800,0.820,0.840,0.860,0.880,0.900,
                  0.910,0.920,0.930,0.940,0.950,0.960,0.970,0.980,0.990,0.999]
        print('CCL Segmentation for multiple thresholds')
        
    else:
        t_list = [threshold]
        print('CCL Segmentation with threshold',threshold)
        
    for t in t_list:
        if not os.path.isdir(save_path+'/'+str(t)):
            os.mkdir(save_path+'/'+str(t))
    
    model.only_edgemap = False
    

    for b, (img, img_name, nms) in enumerate(test_loader):
    
        '''if b==10:
            break'''
        

        # transformed img
        img = img.to('cuda')

        #orig_img = Image.open('./data/bsds/{}.jpg'.format(img_name[0]))

        with torch.no_grad():
            model.eval()
            
            if len(nms.size())==4:
                nms = nms.squeeze().unsqueeze(0)

            height = nms.size()[1]
            width = nms.size()[2]
            
            

            # predict
            _, edges, unaries, edges_c, updated_unaries_c, _, _, _, _, _, _ = model(img,None,shape_dict,contour_cue_dict,nms)
            
            #combined_edges, combined_unaries = edges_c, updated_unaries_c
            combined_edges, combined_unaries = combine_edges([edges,edges_c],[unaries,updated_unaries_c])
            
            #print('Nb of unique pixels before threshold:',len(np.unique(combined_edges)))           
            
            for t in t_list: 
                # use DFS Component Labeling for solving
                # threshold: join prob high enough to count as join
                joins = (combined_unaries[:,0]>=t).tolist()
                joined_edges = list(compress(combined_edges, joins))
                #print('Nb of unique pixels after threshold:',len(np.unique(joined_edges)))
                res = graph_components(joined_edges)
        
                pixels = np.zeros(height*width)
                
        
                for label, component in enumerate(res):
                    
                    for c in component:
                        
                        pixels[c] = label+1
                
                pixels = pixels.reshape(height,width)
                
                name = img_name[0].split('/')[-1]
                
                
                pixels = pixels.convert('RGB')
                pixels.save(save_path+'/'+str(t)+'/'+name+'.png')
                
                # save as binary map:
                #import torchvision
                #torchvision.utils.save_image(torch.tensor(pixels), save_path+'/'+name+'.png')
                
                
                '''
                # save segmentation img
                name = img_name[0].split('/')[-1]
                
                fig, ax = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
        
                ax.imshow(pixels, cmap='jet', alpha=1.0, interpolation=None)
                #ax.set_title(name)
                ax.set_axis_off()
                
                plt.tight_layout()
                plt.savefig(save_path+'/'+name+'.png')
                plt.close(fig)
                '''

# the following two functions are from https://stackoverflow.com/questions/28980797/given-n-tuples-representing-pairs-return-a-list-with-connected-tuples

def dfs(start, graph):
    """
    Does depth-first search, returning a set of all nodes seen.
    Takes: a graph in node --> [neighbors] form.
    """
    visited, worklist = set(), [start]

    while worklist:
        node = worklist.pop()
        if node not in visited:
            visited.add(node)
            # Add all the neighbors to the worklist.
            worklist.extend(graph[node])

    return visited

def graph_components(edges):
    """
    Given a graph as a list of edges, divide the nodes into components.
    Takes a list of pairs of nodes, where the nodes are integers.
    """

    # Construct a graph (mapping node --> [neighbors]) from the edges.
    graph = defaultdict(list)
    nodes = set()

    for v1, v2 in edges:
        nodes.add(v1)
        nodes.add(v2)

        graph[v1].append(v2)
        graph[v2].append(v1)

    # Traverse the graph to find the components.
    components = []

    # We don't care what order we see the nodes in.
    while nodes:
        component = dfs(nodes.pop(), graph)
        components.append(component)

        # Remove this component from the nodes under consideration.
        nodes -= component

    return components
    
        
    

def edge_weight_hist(model, test_loader, shape_dict, contour_cue_dict, model_name):
    save_path = './hists'
    
    
    #model.only_edgemap = False
    
    edge_weights = []
    
    for b, (img, img_name, nms) in enumerate(test_loader):
        

        # transformed img
        img = img.to('cuda')

        #orig_img = Image.open('./data/bsds/{}.jpg'.format(img_name[0]))

        with torch.no_grad():
            model.eval()
            
            if len(nms.size())==4:
                nms = nms.squeeze().unsqueeze(0)

            height = nms.size()[1]
            width = nms.size()[2]
            
            '''
            # graph edge weights
            model.only_edgemap = False
            # predict
            _, edges, unaries, edges_c, updated_unaries_c, _, _, _, _, _, _ = model(img,None,shape_dict,contour_cue_dict,nms)
            
            #combined_edges, combined_unaries = edges_c, updated_unaries_c
            combined_edges, combined_unaries = combine_edges(edges, unaries, edges_c, updated_unaries_c)
            
            edge_weights.append(combined_unaries[:,1].cpu().reshape(-1)) #0=join,1=cut
            '''
            
            # edgemaps
            model.only_edgemap = True
            # predict
            
            res = model(img,nms=nms)
            fuse = res[-1]
            
            edge_weights.append(fuse.squeeze().detach().cpu().numpy().reshape(-1)) 
            
    
    data = np.concatenate(edge_weights).reshape(-1)
    n, bins, patches = plt.hist(data, color='tab:blue',
                            alpha=0.9, weights=np.ones(len(data)) / len(data),bins=100)
    
    plt.xlabel('Edge Weights')
    plt.ylabel('Proportion')
    plt.ylim(0,1)
                            
    plt.savefig(save_path+'/'+model_name+'.png', bbox_inches='tight', pad_inches=0,dpi=120)


if __name__ == '__main__':
    
    ### Args ###
    
    parser = argparse.ArgumentParser(description='Define what you want to predict (BSDS500 data assumed).')
    parser.add_argument('--edges', action='store_true', help='if you want to predict edgemaps')
    parser.add_argument('--multiscale', action='store_true', help='if your edge predictions should use multiscale')
    parser.add_argument('--segs', action='store_true', help='if you want to use KLj segmentation')
    parser.add_argument('--ccl', action='store_true', help='if you want to use connected component labeling segmentation')
    parser.add_argument('--ccl_threshold', default=0.5, help='CCL Threshold where to join, set desired threshold or "multiple" for all thresholds from 0.1 to 0.9')
    parser.add_argument('--model', default='', help='Path to the model checkpoint')
    parser.add_argument('--edgemap_path', default='', help='Where do you want to save the edgemaps? Specify path')
    parser.add_argument('--seg_path', default='', help='Where do you want to save the KL segmentation results? Specify path')
    parser.add_argument('--ccl_path', default='', help='Where do you want to save the CCL segmentation results? Specify path')
    parser.add_argument('--ablation', action='store_true', help='if your model was trained without CRF but with Intervening Contour Cue')
    parser.add_argument('--original_rcf', action='store_true', help='if you only want to predict with original RCF without modification (both backbones available)')
    parser.add_argument('--backbone', default='Resnet', help='Backbone of RCF model (Resnet or VGG)')
    parser.add_argument('--trainset', action='store_true', help='if you want to use the BSDS500 trainset instead of the testset')
    parser.add_argument('--hist', action='store_true', help='make edge weight histograms')
    parser.add_argument('--hist_name', default='hist', help='the name your histogram should have')
    
    args = parser.parse_args()
    
    edges = args.edges
    segmentations = args.segs
    ccl = args.ccl
    
    # path where model is saved
    
    path_to_model = args.model

    # path where you want to save the edgemaps
    if edges:
        
        edgemap_path = args.edgemap_path
        if len(edgemap_path)==0:
            raise Exception('You have to specify where you want to save the edgemaps with the argument --edgemap_path')
         
        mat_path = edgemap_path+'/mat'
        if not os.path.exists(edgemap_path):
            os.mkdir(edgemap_path)
        if not os.path.exists(mat_path):
            os.mkdir(mat_path)
    
    if segmentations:
    
        # path where you want to save the final segmentation results
        
        path_seg = args.seg_path
        if len(path_seg)==0:
            raise Exception('You have to specify where you want to save the segmentation results with the argument --path_seg')
    
        # path where input file for multicut solver will be saved
        # will be deleted after solving (one file around 85 MB)
        path_mc_problem = path_seg+'/mc_problems'
        
        if not os.path.exists(path_seg):
            os.mkdir(path_seg)
        if not os.path.exists(path_mc_problem):
            os.mkdir(path_mc_problem)
            
    if ccl:
        
        ccl_path = args.ccl_path
        if len(ccl_path)==0:
            raise Exception('You have to specify where you want to save the segmentation results with the argument --ccl_path')
        if not os.path.exists(ccl_path):
            os.mkdir(ccl_path)
    
    ################################################################################
    
    print('Load model and data...')
    
    only_rcf = args.ablation
    
    
    # load model
    if args.original_rcf:       
        model = load_model(only_frontend=True, backbone=args.backbone)
        
    else:    
        model = load_model(path_to_model, only_frontend=False, no_crf=only_rcf, backbone=args.backbone)
    
    
    # get data loader (Resnet and VGG need different loaders due to different preprocessing)
    if not args.trainset:
        if args.backbone=='Resnet':
            test_loader = load_bsds_testset(resnet_data_loader)  
        elif args.backbone=='VGG':
            test_loader = load_bsds_testset(vgg_data_loader) 
    else:
        if args.backbone=='Resnet':
            test_loader = load_bsds_trainset(resnet_data_loader)  
        elif args.backbone=='VGG':
        
            test_loader = load_bsds_trainset(vgg_data_loader)

    if edges:
        print('Start predicting edges...')
        if args.multiscale:
            predict_edgemaps_multiscale(model, test_loader, edgemap_path, mat_path)
        else:
            predict_edgemaps(model, test_loader, edgemap_path, mat_path)
        print('Done!')
        
    if ccl:
        try:
            ccl_threshold = float(args.ccl_threshold)
        except:
            # string
            ccl_threshold = args.ccl_threshold
                  
        shape_dict, contour_cue_dict = load_dicts()
        print('Start making segmentations...')
        CCL_segmentation(model, test_loader, shape_dict, contour_cue_dict, ccl_threshold, ccl_path)
        print('Done!')
    
    if segmentations:
        shape_dict, contour_cue_dict = load_dicts()
        print('Start making segmentations...')
        make_segmentation(model, test_loader, shape_dict, contour_cue_dict, path_mc_problem, path_seg)
        print('Done!')
        
    if args.hist:
        print('Making histogram')
        shape_dict, contour_cue_dict = load_dicts()
        edge_weight_hist(model, test_loader, shape_dict, contour_cue_dict, args.hist_name)
        print('Done')