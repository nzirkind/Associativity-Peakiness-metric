"""
This function calculates the associativity - peakiness metric
@author: naomi.e.zirkind
"""
# The argument contmtx is a 2-dimensional real-valued contingency matrix 
# whose rows are truth values and whose columns are cluster numbers.
# The function calculates the associativity-peakiness (AP) metric for this
# contingency matrix.
# Paper that describes theory and performance of the AP metric is at 
# https://doi.org/10.48550/arXiv.2604.22655

 
def calcAPmetric(contmtx):
    
    import numpy as np
    from itertools import combinations
         
    num_truth = contmtx.shape[0]
    
    # Peakiness metric - calculate it for each truth value and then take the mean
    mtx_sorted = np.sort(contmtx, axis=1)  # sort each row along the columns
    row_metrics = np.zeros(num_truth)
    for i in range(num_truth):
        row_metrics[i] = (mtx_sorted[i,-1] - mtx_sorted[i,-2]) / mtx_sorted[i,-1] 
    peakiness_metric = np.mean(row_metrics)
    

    # Associativity metric - metric is highest if max of each column is in a different row        
    col_of_max = np.zeros(num_truth)                
    for i in range(num_truth):
        col_of_max[i] = np.argmax(contmtx[i,:])    
    col_of_max_list = col_of_max.tolist()
    pairs_list = list(combinations(col_of_max_list, 2))
    numsame = 0
    for element in pairs_list:
        if (element[0] == element[1]):
            numsame +=1
    associativity_metric = (len(pairs_list) - numsame) / len(pairs_list)  
    
    
    # AP metric is harmonic mean of peakiness and associativity metrics
    AP_metric = 2 * (peakiness_metric * associativity_metric) / \
        (peakiness_metric + associativity_metric)
    
    return(AP_metric)  
      
