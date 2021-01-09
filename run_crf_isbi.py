import argparse
import networkx as nx
import itertools
import torch
import torch.nn as nn
import numpy as np
import pickle
import h5py
import subprocess

from src.utils.crf import get_graph_idx
from src.cycle_monitoring.cycles import check_cycles
from src.models.crf_multicut import CrfMulticut
from src.utils.rcf_crf import combine_edges


def preprocess_edges(edges):

    nb_edges = len(edges)

    # build graph
    g = nx.Graph()
    g.add_edges_from(edges)

    # find all 3-cliques (for few edges)
    cliques = nx.find_cliques(g)    
    cliques3 = list(set(sum([list(itertools.combinations(set(clq), 3)) for clq in cliques if len(clq)>=3],[])))
    
    # different approach that works for more edges
    #cliques3 = [c for c in nx.cycle_basis(g) if len(c)==3]
    
    print('Graph has '+str(g.number_of_edges())+' edges and '+str(len(cliques3))+' 3-cliques')

    edge_index, j_edge_indices, cliques_ints = get_graph_idx(edges, cliques3)

    return edge_index, j_edge_indices, cliques_ints

def node_ids_to_edges(local_u_ids,local_v_ids,lifted_u_ids,lifted_v_ids):

    edges = np.stack([local_u_ids,local_v_ids],1)
    edges = edges.tolist()
    edges = [tuple(e) for e in edges]

    lifted_edges = np.stack([lifted_u_ids,lifted_v_ids],1)
    lifted_edges = lifted_edges.tolist()
    lifted_edges = [tuple(e) for e in lifted_edges]

    return edges, lifted_edges

def make_lifted_mc_problem(nb_nodes, edges, unaries, nb_neighbors_edges, path):

    outputs = unaries[:,0] #0=join, 1=cut

    nb_lifted_edges = len(edges)-nb_neighbors_edges
    
    # combine edges and outputs
    combined = np.column_stack((edges,outputs))

    #print('Start creating file.')

    # create txt file from data
    filename=path
    np.savetxt(filename,combined, fmt='%d %d %.10f',header=str(nb_nodes)+'\n'+str(nb_neighbors_edges)+'\n'+str(nb_lifted_edges),comments='')

    return filename



if __name__ == '__main__':

    # run with Python 3
    
    ### Args ###
    
    parser = argparse.ArgumentParser(description='Define paths.')
    parser.add_argument('--saved_prob_model', help='Path to the saved model containing edges and probabilities.', default='ISBI2012/ISBI2012/code/data/NaturePaperDataUpl/ISBI2012_output_folder/prob_model.h5')
    parser.add_argument('--saved_energy_model', help='Path to the saved model containing edges and energies.', default='ISBI2012/ISBI2012/code/data/NaturePaperDataUpl/ISBI2012_output_folder/energy_model.h5')
    parser.add_argument('--out_path', help='Path to output directory', default='ISBI2012/ISBI2012/code/data/NaturePaperDataUpl/ISBI2012_output_folder/')
    parser.add_argument('--solve_KLj', action='store_true', help='if specified KLj will be used to solve, else you use the CRF and afterwards use the saved updated probs to use the original solver by running the file /ISBI2012/ISBI2012/code/run_mc_2d_crf.py with --use_lifted True or transform to energies and then run this script again with --solve_KLj')
    
    args = parser.parse_args()
    
    
    if not args.solve_KLj:
    
        prob_filename = args.saved_prob_model
    
        with h5py.File(prob_filename, "r") as f:
            # List all groups
            #print("Keys: %s" % f.keys())
            print('Reading ',prob_filename)
            # Get the data
            lifted_probs = np.array(f['lifted_probs'])
            lifted_u_ids = np.array(f['lifted_u_ids'])
            lifted_v_ids = np.array(f['lifted_v_ids'])
            local_probs = np.array(f['local_probs'])
            local_u_ids = np.array(f['local_u_ids'])
            local_v_ids = np.array(f['local_v_ids'])
            number_of_vertices = np.array(f['number-of-vertices'])
    
        
        edges, lifted_edges = node_ids_to_edges(local_u_ids,local_v_ids,lifted_u_ids,lifted_v_ids)
        
        # only local edges for now, lifted have to be subsampled first as there would be too many cliques
        edge_index, j_edge_indices, cliques_ints = preprocess_edges(edges)
        #edge_index, j_edge_indices, cliques_ints = preprocess_edges(lifted_edges)
    
        # make probs to tensor
        local_unaries = torch.tensor([local_probs,1-np.array(local_probs)]).t()
    
        # CRF
        print('Total nb of cycles: ',len(cliques_ints))
        print('Before CRF:')
        cycle_counts_rounded_before = check_cycles(local_unaries, cliques_ints,rounded=True)
        cycle_counts_before = check_cycles(local_unaries, cliques_ints,rounded=False)
        print('Number of invalid cycles (rounded): ',cycle_counts_rounded_before[1])
        print('Number of invalid cycle inequalities: ',cycle_counts_before[1])
        
        if args.basic_crf:
            print('Using basic CRF without exponent...')
            crf = CrfMulticut(20, True, True, 1)
        else:
            print('Using CRF with exponent update during mean field update...')
            crf = CrfMulticut(20,True,True,1,postprocess=True)
        crf.eval()
        crf.cost_valid = nn.Parameter(torch.tensor([0.001]))
        crf.cost_invalid = nn.Parameter(torch.tensor([0.999]))
        print('Cost for valid cycles: ',crf.cost_valid)
        print('Cost for invalid cycles: ',crf.cost_invalid)
    
    
        updated_local_unaries, cycle_counts_rounded, cycle_counts = crf(local_unaries, edge_index,j_edge_indices,cliques_ints)
    
        print('Number of invalid cycles (rounded): ',cycle_counts_rounded[1])
        print('Number of invalid cycle inequalities: ',cycle_counts[1])
    
        # save
        updated_local_probs = updated_local_unaries.detach().numpy()[:,0]
        save_path = args.out_path+"pTest_local_crf.pkl"
        pickle.dump(updated_local_probs, open(save_path,"wb"), protocol=2)
    
        print('Updated local probabilities have been saved to ',save_path)

    
    else:

        filename = args.saved_energy_model
    
        with h5py.File(filename, "r") as f:
            # List all groups
            #print("Keys: %s" % f.keys())
            print('Reading ',filename)
            # Get the data
            lifted_energies = list(f['lifted_energies'])
            lifted_u_ids = list(f['lifted_u_ids'])
            lifted_v_ids = list(f['lifted_v_ids'])
            local_energies = list(f['local_energies'])
            local_u_ids = list(f['local_u_ids'])
            local_v_ids = list(f['local_v_ids'])
            number_of_vertices = list(f['number-of-vertices'])
                 
        
        lifted_energies = torch.tensor([lifted_energies,1-np.array(lifted_energies)]).t()
        local_energies = torch.tensor([local_energies,1-np.array(local_energies)]).t()
        edges, lifted_edges = node_ids_to_edges(local_u_ids,local_v_ids,lifted_u_ids,lifted_v_ids)
        combined_edges, combined_unaries = combine_edges([edges,lifted_edges],[local_energies,lifted_energies])
        nb_nodes = len(number_of_vertices)
        nb_neighbors_edges = len(edges)
        path=args.out_path+'mc_problem_isbi.txt'
        
        print('Creating multicut problem txt file')
        mc_problem = make_lifted_mc_problem(nb_nodes, combined_edges, combined_unaries, nb_neighbors_edges, path)
        
        print('Solving...')
        
        filename_out = args.out_path+'isbi_CRF_KLj_result.h5'
        subprocess.call(['./graph_lib/graph/build/solve-regular-lifted-energies', '-i', mc_problem, '-o', filename_out])
        
        print('Solving finished!')
        
        # Load labels from newly created h5 file
        with h5py.File(filename_out, "r") as f:
            labels = np.array(f['labels'])
            
        # save
        pickle.dump(labels, open(args.out_path+"KLj_labels_isbi.pkl","wb"), protocol=2)
        print("Results saved in provided output directory")