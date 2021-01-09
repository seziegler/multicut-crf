import itertools
import os
import time
from os.path import isdir, join
import numpy as np
import torch
import torchvision

from src.utils.rcf import Averagvalue


def train(train_loader, model, rcf_loss, optimizer, scheduler, epoch, save_dir,shape_dict={}, contour_cue_dict={}, itersize=10, print_freq=100, maxepoch=20, backbone='Resnet', method='power'):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()

    # switch to train mode
    model.only_edgemap = False
    model.train()

    end = time.time()
    epoch_loss = []
    counter = 0

    # init cycle counts
    cycle_counts_rounded_list = []
    cycle_counts_list = []

    for i, (image, label, img_name, nms) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image, label = image.cuda(), label.cuda()

        rcf_outputs, edges, unaries, edges_c, unaries_c, cliques, seg_labels, cycle_counts_rounded, cycle_counts, shape_dict, contour_cue_dict = model(image,label, shape_dict, contour_cue_dict, nms)

        if backbone=='VGG':
            loss = torch.zeros(1).cuda()
            for o in rcf_outputs:
                loss = loss + rcf_loss(o, label)

        elif backbone=='Resnet':
            # only fuse loss as in https://github.com/mayorx/rcf-edge-detection
            loss = rcf_loss(rcf_outputs[-1], label)


        counter += 1
        loss = loss / itersize

        # crf loss
        # seg_labels: Cut/Join Ground Truth created while training
        # unaries_c: predicted cut probability for all edges that form a 3-cliques

        seg_labels = torch.tensor(seg_labels,dtype=torch.long).to('cuda')
        cut_loss = rcf_loss(unaries_c,seg_labels)

        # combine edge loss and cut loss
        loss = loss + cut_loss


        loss.backward()
        if counter == itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        cycle_counts_rounded_list.append(cycle_counts_rounded)
        cycle_counts_list.append(cycle_counts)

        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if (i+1) % print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch+1, maxepoch, i+1, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            label_out = torch.eq(label, 1).float()
            rcf_outputs.append(label_out)
            _, _, H, W = rcf_outputs[0].shape
            all_results = torch.zeros((len(rcf_outputs), 1, H, W))
            for j in range(len(rcf_outputs)):
                all_results[j, 0, :, :] = rcf_outputs[j][0, 0, :, :]
            torchvision.utils.save_image(all_results, join(save_dir, "iter-%d.jpg" % i))
        # save checkpoint
    '''save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
            }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))'''

    # invalid cycles
    print('Epoch finished!')
    vals_r,invals_r = zip(*cycle_counts_rounded_list)
    vals,invals = zip(*cycle_counts_list)
    print('Avg number of invalid cycles (rounded): ',np.mean(invals_r))
    print('Avg number of invalid cycle inequalities: ',np.mean(invals))


    # increase p for further CRF optimization if inequalities are already very low
    if np.mean(invals) < 1:
        if method=='power':
            model.crf.p += 0.1
            #model.crf.p += pow(model.crf.p,-1)
        elif method=='factor':
            model.crf.p += 1
        optimizer, scheduler = reset_opt(optimizer, scheduler, backbone)
        print('CRF parameter p has been increased to',model.crf.p)

    return losses.avg, epoch_loss, shape_dict, contour_cue_dict


def reset_opt(optimizer, scheduler, backbone='Resnet'):
    scheduler._last_lr = scheduler.base_lrs

    if backbone == 'Resnet':

        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['initial_lr']

    else:
        nb_param_groups = len(optimizer.param_groups)

        for par in range(nb_param_groups):
            optimizer.param_groups[par]['lr'] = optimizer.param_groups[par]['initial_lr']

    return optimizer, scheduler


def intervening_contour_cue(edge_map, edge_gt, k, prior=0.45, contour_cue_dict={}, dict_update=True):

    # k = neighborhood size (how far are pixels away from source pixel)

    assert(len(edge_map.size())==4), 'Edge Map dimensions not as expected! \nGiven: {} \nExpected: torch.Size([1,1,height,width])'.format(edge_map.size())

    # check device type
    dev = edge_map.get_device()
    if dev == -1:
        dev = 'cpu'

    prior = torch.tensor(prior)
    prior = prior.to(dev)

    # init dicts, one holds all edges that form cliques, the other one holds remaining edges
    edge_features_list = []
    edge_features_cliques_list = []

    # init cliques list
    cliques = []

    # generate labels for every edge_feature that is in clique
    #seg_mask = seg_mask[0] # first dim is batch
    labels_list = []

    height = edge_map.size()[2]
    width = edge_map.size()[3]
    # enumerate pixels
    pixels = torch.arange(height*width).view(height,width)

    # init lists for saving coords of edges so that next time computation is faster
    edges_coords_list = []
    edges_cliques_coords_list = []
    # edges that are neighbors, not in a clique and get default value anyway
    edges_only_neighbors = {}

    em = edge_map[0,0]
    shape = edge_map.size()

    if edge_gt != None:
        edge_gt = edge_gt[0,0]
        #print(edge_gt)
        #print(edge_gt.size())
        #edge_gt = torch.tensor(edge_gt).to(dev)

    # bottom right corner pixel
    #edge_features[int(pixels[height-1,width-1]),int(pixels[height-2,width-1])] = prior
    edges_only_neighbors[int(pixels[height-1,width-1]),int(pixels[height-2,width-1])] = prior
    #edge_features[(height-1,width-1),(height-1,width-2)] = prior

    # minimal distance between pixles for computing the edge
    start_k = 2 # no direct neighbors

    for k_d in range(start_k,k+1):

        # create new dict that holds edge coords and labels for specific k
        # (so that every row has same length and they can later be transforemd to a tensor)

        # all edges that cover distance k but are not in a clique
        edges_coords = {}
        # edges that form clique
        edges_cliques_coords = {}

        # init dicts, one holds all edges that form cliques, the other one holds remaining edges
        edge_features = {}
        edge_features_cliques = {}

        labels = []

        for i in range(0,height-1):
            print(k_d, i)
            for j in range(0,width-1):

                if k_d==start_k:# first run, include direct neighbors

                    ### direct neighbors that get prior prob
                    ## upper neighbor
                    #up_n = i,j-1
                    #edge_features[(i,j),(up_n[0],up_n[1])] = prior
                    ## lower neighbor
                    low_n = i,j+1
                    #edge_features[int(pixels[i,j]),int(pixels[low_n[0],low_n[1]])] = prior
                    edges_only_neighbors[int(pixels[i,j]),int(pixels[low_n[0],low_n[1]])] = prior
                    ## left neighbor
                    #left_n = i-1,j
                    #edge_features[(i,j),(left_n[0],left_n[1])] = prior
                    ## right neighbor
                    right_n = i+1,j
                    #edge_features[int(pixels[i,j]),int(pixels[right_n[0],right_n[1]])] = prior
                    edges_only_neighbors[int(pixels[i,j]),int(pixels[right_n[0],right_n[1]])] = prior

                if i >= k_d  and j < width-k_d:

                    ### get positions of other pixels
                    # left upper corner
                    #lu = i-k,j-k
                    # top middle
                    tm = i-k_d,j
                    # right upper corner
                    ru = i-k_d,j+k_d
                    # right middle
                    rm = i,j+k_d
                    # left middle
                    #lm = i, j-k
                    # bottom left corner
                    #bl = i+k,j-k
                    # bottom middle
                    bm = i+k_d,j
                    # bottom right corner
                    #br = i+k,j+k

                    ### get maximum on the line between source and target pixel
                    # tm
                    if j>=k_d:
                        edge_features_cliques[int(pixels[i,j]),int(pixels[tm[0],tm[1]])] = torch.max(em[tm[0]:i+1,tm[1]])
                        edges_cliques_coords[int(pixels[i,j]),int(pixels[tm[0],tm[1]])] = np.stack([np.arange(tm[0],i+1),[tm[1]]*(k_d+1)])

                        if edge_gt != None:
                            # labels based on edge_gt
                            labels.append(1 if torch.max(edge_gt[tm[0]:i+1,tm[1]]) >= 0.5 else 0)
                    else:
                        edge_features[int(pixels[i,j]),int(pixels[tm[0],tm[1]])] = torch.max(em[tm[0]:i+1,tm[1]])
                        edges_coords[int(pixels[i,j]),int(pixels[tm[0],tm[1]])] = np.stack([np.arange(tm[0],i+1),[tm[1]]*(k_d+1)])

                    # rm
                    if j < width-2*k_d:
                        edge_features_cliques[int(pixels[i,j]),int(pixels[rm[0],rm[1]])] = torch.max(em[rm[0],j:rm[1]+1])
                        edges_cliques_coords[int(pixels[i,j]),int(pixels[rm[0],rm[1]])] = np.stack([[rm[0]]*(k_d+1),np.arange(j,rm[1]+1)])

                        if edge_gt != None:
                            # labels based on edge_gt
                            labels.append(1 if torch.max(edge_gt[rm[0],j:rm[1]+1]) >= 0.5 else 0)
                    else:
                        edge_features[int(pixels[i,j]),int(pixels[rm[0],rm[1]])] = torch.max(em[rm[0],j:rm[1]+1])
                        edges_coords[int(pixels[i,j]),int(pixels[rm[0],rm[1]])] = np.stack([[rm[0]]*(k_d+1),np.arange(j,rm[1]+1)])

                    # ru
                    line = [(i-s,j+s) for s in range(0,k_d+1)]
                    if j < width-2*k_d:
                        edge_features_cliques[int(pixels[i,j]),int(pixels[ru[0],ru[1]])] = torch.max(em[tuple(np.array(line).T)])
                        edges_cliques_coords[int(pixels[i,j]),int(pixels[ru[0],ru[1]])] = np.array(line).T.squeeze().reshape(2,-1)#np.array(line).T.squeeze()

                        if edge_gt != None:
                            # labels based on edge_gt
                            labels.append(1 if torch.max(edge_gt[tuple(np.array(line).T)]) >= 0.5 else 0)
                    else:
                        edge_features[int(pixels[i,j]),int(pixels[ru[0],ru[1]])] = torch.max(em[tuple(np.array(line).T)])
                        edges_coords[int(pixels[i,j]),int(pixels[ru[0],ru[1]])] = np.array(line).T.squeeze().reshape(2,-1)#np.array(line).T.squeeze()


                    # get cliques
                    if j<width-2*k_d:
                        cliques.append((int(pixels[i,j]),int(pixels[rm]),int(pixels[ru])))
                    # if uncommented each edge will be in different amount of cliques (less efficient computation in later steps)
                    #if i<height-2*k and j<width-2*k and j>=k:
                        #cliques.append((int(pixels[bm]),int(pixels[i,j]),int(pixels[rm])))


        # append coords, results for this k to respective lists

        edge_features_list.append(edge_features)
        edge_features_cliques_list.append(edge_features_cliques)
        edges_coords_list.append(edges_coords)
        edges_cliques_coords_list.append(edges_cliques_coords)
        labels_list.append(labels)


    # get number of k's
    nb_ks = len(edge_features_list)

    # direct neighbors
    edges_n = list(edges_only_neighbors.keys())
    unaries_n = list(edges_only_neighbors.values())
    unaries_n = torch.stack(unaries_n)
    unaries_n = torch.stack((1-unaries_n,unaries_n),dim=1)


    # init final lists that hold egdes, coords and unaries for each k
    all_edges = []
    all_unaries = []
    all_edges_c = []
    all_unaries_c = []
    all_labels = [] # labels only for edges that form cliques
    all_edges_coords = []
    all_edges_cliques_coords = []


    for i in range(nb_ks):
        #print(edge_features_list[i])

        # edges that are not in cliques
        edges = list(edge_features_list[i].keys())
        unaries = list(edge_features_list[i].values())
        unaries = torch.stack(unaries)
        unaries = torch.stack((1-unaries,unaries),dim=1)

        # edges that form cliques
        edges_c = list(edge_features_cliques_list[i].keys())
        unaries_c = list(edge_features_cliques_list[i].values())
        unaries_c = torch.stack(unaries_c)
        unaries_c = torch.stack((1-unaries_c,unaries_c),dim=1)
        if edge_gt != None:
            k_labels = torch.tensor(labels_list[i],dtype=torch.long)

        # save dict
        k_edges_coords = list(edges_coords_list[i].values())
        k_edges_coords = torch.tensor(k_edges_coords,dtype=torch.long)

        k_edges_cliques_coords = list(edges_cliques_coords_list[i].values())
        k_edges_cliques_coords = torch.tensor(k_edges_cliques_coords,dtype=torch.long)

        # append
        all_edges.append(edges)
        all_unaries.append(unaries)
        all_edges_c.append(edges_c)
        all_unaries_c.append(unaries_c)
        if edge_gt != None:
            all_labels.append(k_labels)
        all_edges_coords.append(k_edges_coords)
        all_edges_cliques_coords.append(k_edges_cliques_coords)


    # combine results
    all_edges_out = list(itertools.chain.from_iterable(all_edges))
    all_unaries_out = torch.cat(all_unaries,dim=0)

    all_edges_c_out = list(itertools.chain.from_iterable(all_edges_c))
    all_unaries_c_out = torch.cat(all_unaries_c,dim=0)

    if edge_gt != None:
        #all_labels_out = list(itertools.chain.from_iterable(all_labels))
        all_labels_out = torch.cat(all_labels,dim=0)
    else:
        all_labels_out = None

    all_edges_coords_out = list(itertools.chain.from_iterable(all_edges_coords))
    all_edges_cliques_coords_out = list(itertools.chain.from_iterable(all_edges_cliques_coords))

    if dict_update:
        # dict for fast version
        contour_cue_dict[shape] = [all_edges_coords,all_edges_cliques_coords,edges_only_neighbors,cliques,all_edges_out,all_edges_c_out]

    return all_edges_out, all_unaries_out, all_edges_c_out, all_unaries_c_out, cliques, all_labels_out, contour_cue_dict


def intervening_contour_cue_fast(edgemap, edge_gt, contour_cue_dict):
    # load coords from dictionary for given shape
    edges_coords, edges_cliques_coords, edges_only_neighbors, cliques, edges, edges_c = contour_cue_dict[edgemap.size()]

    edgemap = edgemap[0, 0]

    # get max values for each edge
    unaries = [get_unaries(edgemap, coords) for coords in edges_coords]
    unaries = torch.cat(unaries, dim=0)

    unaries_c = [get_unaries(edgemap, coords) for coords in edges_cliques_coords]
    unaries_c = torch.cat(unaries_c, dim=0)

    if edge_gt != None:  # is None when testing
        edge_gt = edge_gt[0, 0]
        # same for labels but using ground truth (only for edges that are part of cliques)
        labels = [get_labels(edge_gt, coords) for coords in edges_cliques_coords]
        labels = torch.cat(labels, dim=0)
    else:
        labels = None

    # make type long

    return edges, unaries, edges_c, unaries_c, cliques, labels, contour_cue_dict


def get_unaries(edgemap,edges_coords):

    unaries = edgemap[(edges_coords[:,0],edges_coords[:,1])].max(1)[0]
    unaries = torch.stack((1-unaries,unaries),dim=1)

    return unaries


def get_labels(edge_gt,edges_coords):

    labels = edge_gt[(edges_coords[:,0],edges_coords[:,1])].max(1)[0]

    # no annotator marked the pixel as edge point
    labels[labels==0] = 0
    # less than half of the annotators (but at least one) marked the pixel as edge point -> controversial
    labels[np.logical_and(labels>0,labels<0.5)] = 2
    # more than half of the annotators marked the pixel as edge point
    labels[labels>=0.5] = 1

    return labels


def combine_edges(edge_lists, tensor_lists):

    combined_edge_lists = list(itertools.chain(*edge_lists))
    combined_unaries = torch.cat(tensor_lists)

    return combined_edge_lists, combined_unaries


def mask_nms(edgemap, nms):

    nms2 = torch.zeros(nms.size())
    nms2[nms] = 1.0

    # set all values where there is no edge according to precomputed NMS to 0
    edgemap_thin = edgemap*nms2.to('cuda')

    return edgemap_thin


def create_lifted_mc_problem(size, edges, unaries, nb_neighbors_edges, img_name, path):

    outputs = unaries[:,0]  # 0=join, 1=cut

    height, width = size

    nb_nodes = height*width
    nb_edges = len(edges)

    # adapt indices for multicut library
    new_idx = np.arange(0,nb_nodes,1)
    e = list(zip(*edges))[0]+list(zip(*edges))[1]
    pixels = [new_idx[v] for v in set(e)]
    unique_coords = list(set(e))
    coord_pixel_mapping = dict(zip(unique_coords, pixels))
    mapped_edges = [(coord_pixel_mapping.get(coord1,coord1),coord_pixel_mapping.get(coord2,coord2)) for coord1, coord2 in edges]

    # combine edges and outputs
    combined = np.column_stack((mapped_edges,outputs.cpu()))

    # create txt file from data
    filename=path+'/mc_problem_{}.txt'.format(img_name[0].split('/')[-1])
    np.savetxt(filename,combined, fmt='%d %d %.10f',header=str(nb_nodes)+'\n'+str(nb_neighbors_edges)+'\n'+str(nb_edges),comments='')

    return filename