from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score
#from sklearn.utils import linear_assignment_
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.metrics import variation_of_information


def VOI(img_gt, img_labeled):
    """
    Variation of Information
    """
    
    # gives conditional entropies H(X|Y) and H(Y|X)
    split_vi, merge_vi = variation_of_information(img_gt,img_labeled)
    
    # VI(X,Y) = H(X|Y) + H(Y|X)
    return split_vi+merge_vi, split_vi, merge_vi
       

class NMI():
    """
    normalized_mutual_info_score for cluster assignment
    """

    def __init__(self):
        pass

    def __call__(self, outputs, target):

        self.res = normalized_mutual_info_score(target, outputs)
        return self.value()

    def value(self):

        return self.res

    def name(self):
        return 'Normalized Mutual Info Score'


class AdjustedRand():
    """
    adjusted_rand_score for cluster assignment
    """

    def __init__(self):
        pass

    def __call__(self, outputs, target):
        self.res = adjusted_rand_score(target, outputs)
        return self.value()

    def value(self):
        return self.res

    def name(self):
        return 'Adjusted Rand Score'


# from https://github.com/herandy/DEPICT/blob/e75840d7b401b919d00abfb905169e02e40daba2/functions.py#L318
class BestMap():

    """
    Accuracy for cluster assignment using Hungarian Algorithm
    """

    def __init__(self):
        pass

    def __call__(self, L2, L1):

        if L1.__len__() != L2.__len__():
          print('size(L1) must == size(L2)')

        Label1 = np.unique(L1)
        nClass1 = Label1.__len__()
        Label2 = np.unique(L2)
        nClass2 = Label2.__len__()

        nClass = max(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            for j in range(nClass2):
                G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

        c = np.array(list(zip(*linear_sum_assignment(-G.T))))[:, 1]

        newL2 = np.zeros(L2.__len__())
        for i in range(nClass2):
            for j in np.nonzero(L2 == Label2[i])[0]:
                if len(Label1) > c[i]:
                    newL2[j] = Label1[c[i]]
        self.res = accuracy_score(L1, newL2)
        return self.value()

    def value(self):
        return self.res

    def name(self):
        return 'Best Map'

