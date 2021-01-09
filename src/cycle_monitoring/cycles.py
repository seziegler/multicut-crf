
import torch

def check_cycles(cur_U, cliques_ints, rounded=False):
    
    # check device type
    dev = cur_U.get_device()
    if dev == -1:
        dev = 'cpu'

    #### no gradient computation here
    cur_U_detached = cur_U.clone().detach()

    # only look at cut probability
    cut_probs = cur_U_detached[cliques_ints][:,:,1] # 1=cut, 0=join
    
    if rounded:
    
        ### rounded 
        # round and sum every clique
        rounded_sums = cut_probs.round().sum(dim=1)
        # check if sum is 1 (only possible in invalid case (1,0,0))
        nb_invalid_cycles_rounded = len(rounded_sums[rounded_sums==1])

        #print('Number of rounded invalid cycles: '+str(nb_invalid_cycles_rounded)+'/'+str(len(cliques_ints)))
        
        # return number of valid, invalid cliques
        return [len(cliques_ints)-nb_invalid_cycles_rounded,nb_invalid_cycles_rounded]
    
    else:
    
        ### not rounded (cycle inequalities)
        # get max value of each clique and corresponding position within the clique
        max_vals, ind = cut_probs.max(dim=1)

        # based on previously obtained position of max value, sum the other two values
        m,n = cut_probs.shape
        sum_smaller_vals = cut_probs[torch.arange(n, device=dev) != ind[:,None]].reshape(m,-1).sum(dim=1)

        # check whether sum of the other two values is smaller than max value for every clique
        nb_cycle_inequalities = int(torch.sum(max_vals > sum_smaller_vals))

        #print('Number of cycle inequalities: '+str(nb_cycle_inequalities)+'/'+str(len(cliques_ints)))
        
        # return number of valid, invalid cliques
        return [len(cliques_ints)-nb_cycle_inequalities,nb_cycle_inequalities]