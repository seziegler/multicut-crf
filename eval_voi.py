import argparse
import numpy as np
from scipy.io import loadmat
import os

from src.evaluation.metrics import VOI


if __name__ == '__main__':
    
    ### Args ###
    
    parser = argparse.ArgumentParser(description='Define which segmentations you want to evaluate.')
    parser.add_argument('--seg_path', help='Path to the segmentations saved as .mat')
    parser.add_argument('--gt_path', help='Path to the Ground Truth directory', default='data/bsds/test/gt_seg')
    
    
    args = parser.parse_args()
    
    seg_path = args.seg_path
    GT_path = args.gt_path
    
    vi_list = []
    vi_split_list = []
    vi_merge_list = []
    
    for (dirpath, dirnames, filenames) in os.walk(seg_path):
        for file_name in filenames:
            
            seg = loadmat(seg_path+'/'+file_name)
            #print(seg)
            seg = seg['seg'].astype(np.int)
            
            # load ground truth
            gt = loadmat(GT_path+'/'+file_name)
            gt = gt['groundTruth']
            num_gts = gt.shape[1]
            GTs = [gt[0,i]['Segmentation'][0,0].astype(np.int32) for i in range(num_gts)]
            
            # one img has many ground truths    
            vi_img_list = []
            vi_split_img_list = []
            vi_merge_img_list = []
            for g_seg in GTs:
                           
                vi, vi_split, vi_merge = VOI(g_seg,seg)
                vi_img_list.append(vi)
                vi_split_img_list.append(vi_split)
                vi_merge_img_list.append(vi_merge)
            
            vi_img = np.mean(vi_img_list)
            vi_split_img = np.mean(vi_split_img_list)
            vi_merge_img = np.mean(vi_merge_img_list)
            
            vi_list.append(vi_img)
            vi_split_list.append(vi_split_img)
            vi_merge_list.append(vi_merge_img)
            
            #print('VOI for {}: '.format(file_name),vi)
    
    vi_all = np.mean(vi_list)
    vi_split_all = np.mean(vi_split_list)
    vi_merge_all = np.mean(vi_merge_list)
    
    print('Overall VOI: ',vi_all)
    print('Split VOI: ',vi_split_all)
    print('Merge VOI: ',vi_merge_all)
        