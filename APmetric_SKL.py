# -*- coding: utf-8 -*-



# Paper that describes theory and performance of metric is at 
# https://doi.org/10.48550/arXiv.2604.22655



def associativity_peakiness_AP_score(contmtx):
    """
    Computes the associativitiy, peakiness, and AP_metric scores in a single 
    function.
    
    Those metrics are based on direct analysis of the values in a 
    contingency table whose row labels are Ground Truth class labels and 
    column labels are indices of the clusters formed by a clustering algorithm.
     
    For each row in the contingency table, identify the column which has the largest
    value in that row. The result is a matching between clusters and truth labels.
    
    A contingency table satisfies associativity if this matching is one-to-one, i.e. 
    only one cluster is associated with each truth label. The associativity is 
    measured by, for each row of the econtingency table, determining the index of the
    cluster with the largest value in that row. A list is formed containing the
    cluster indices selected for each row. Pairs are formed between each of these
    elements, and then pairs whose elements are equal are discarded from the list. 
    The associativity is the ratio of the length of the resulting list to the length
    of the original list.
    
    A contingency table satisfies peakiness if the largest value in each row is much
    larger than all the other elements in that row. The peakiness score is equal to
    the mean, over all rows of the contingency table, of the ratio of the 
    difference between the two largest elements in a row, and the largest element
    in that row.
    
    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.
    
    Read more in the :ref:`User Guide <associativity-peakiniess>`.
    
    Parameters
    ----------
    contmtx : array-like of shape (n_truth_classes, n_clusters)
        Ground truth class labels are the rows, and cluster indices are the columns.
    
        
    Returns
    -------
    associativity : float
        Score between 0.0 and 1.0. 1.0 stands for one-to-one matching between truth
        labels and clusters.
    
    peakiness : float
        Score between 0.0 and 1.0. 1.0 stands for a contingency table for which
        all elements in each row, besides the peak value, are equal to 0.
    
    AP_metric : float
        Harmonic mean of the associativity and the peakiness.
    
    
    See Also
    --------
    associativity_metric : Associativity between truth labels and clusters in a contingency table.
    peakiness_metric : Peakiness of the peak values in rows of a contingency table.
    AP_metric : AP metric, the harmonic mean of associativity and peakiness metric scores.
    
    Examples
    --------
    >>> from sklearn.metrics import associativity_peakiness_AP_score
    >>> contmtx = np.array([[4, 2, 1, 1],
                            [1, 6, 0, 1],
                            [1, 2, 2, 3]])
    
    >>> associativity_peakiness_AP_score(contmtx)
    (1.0, 0.556, 0.714)
    
    
    References
    ----------

    .. [1] `Naomi E Zirkind and William J Diehl, 2026. Associativity-Peakiness 
            Metric for Contingency Tables
       <https://doi.org/10.48550/arXiv.2604.22655f>`_

    
    
    
    """

     
   
    import numpy as np
    from itertools import combinations
         
    if len(contmtx.shape) != 2:
        print('Invalid input. Contingency matrix must have 2 dimensions.')
        return
    
    
    num_truth = contmtx.shape[0]
    
    # Peakiness metric - calculate it for each truth value and then take the mean
    mtx_sorted = np.sort(contmtx, axis=1)  # sort each row along the columns
    row_metrics = np.zeros(num_truth)
    for i in range(num_truth):
        if mtx_sorted[i,-1] == 0:
            row_metrics[i] = 0
        else:
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
    if peakiness_metric + associativity_metric == 0:
        AP_metric = 0
    else:
        AP_metric = 2 * (peakiness_metric * associativity_metric) / \
            (peakiness_metric + associativity_metric)
    
    return float(associativity_metric), float(peakiness_metric), float(AP_metric)   




def associativity_metric(contmtx):
    
    """Associativity metric of a contingency table given a ground truth.

    For each row in the contingency table, identify the column which has the largest
    value in that row. The result is a matching between clusters and truth labels.
    A contingency table satisfies associativity if this matching is one-to-one, i.e. 
    only one cluster is associated with each truth label. 

    Read more in the :ref:`User Guide <associativity-peakiness>`.

    Parameters
    ----------
    contmtx : array-like of shape (n_truth_classes, n_clusters)
        Ground truth class labels are the rows, and cluster indices are the columns.

    Returns
    -------
    associativity_metric : float
        Score between 0.0 and 1.0. 1.0 stands for one-to-one matching between truth
        labels and clusters.

    See Also
    --------
    peakiness_metric : Peakiness of the peak values in rows of a contingency table.
    AP_metric : AP metric, the harmonic mean of associativity and peakiness metric scores.

    References
    ----------

    .. [1] `Naomi E Zirkind and William J Diehl, 2026. Associativity-Peakiness 
            Metric for Contingency Tables
       <https://doi.org/10.48550/arXiv.2604.22655f>`_

    Examples
    --------

    one-to-one labelings are associative:

      >>> from sklearn.metrics.cluster import associativity_metric
      >>> contmtx = np.array([[4, 2, 1, 1],
                              [1, 6, 0, 1],
                              [1, 2, 2, 3]])
      >>>associativity_metric(contmtx)
      1.0

    For each row, the peak value is in a different column.
    """
        
    return associativity_peakiness_AP_score(contmtx)[0]


def peakiness_metric(contmtx):
    
    """Compute peakiness metric of a contingency table given a ground truth.

    A contingency table satisfies peakiness if the largest value in each row is 
    much larger than all the other elements in that row. 

    Read more in the :ref:`User Guide <associativity-peakiness>`.

    Parameters
    ----------
    contmtx : array-like of shape (n_truth_classes, n_clusters)
        Ground truth class labels are the rows, and cluster indices are the columns.
        
        
    Returns
    -------
    peakiness_metric : float
        Score between 0.0 and 1.0. 1.0 stands for complete peakiness of peak 
        value in each row of the contingency table, i.e., all elements besides 
        peak value are 0.

    See Also
    --------
    associativity_metric : Associativity between truth labels and clusters in a contingency table.
    AP_metric : AP metric, the harmonic mean of associativity and peakiness metric scores.

    References
    ----------

    .. [1] `Naomi E Zirkind and William J Diehl, 2026. Associativity-Peakiness 
            Metric for Contingency Tables
       <https://doi.org/10.48550/arXiv.2604.22655f>`_

    Examples
    --------

    Ideally peaky rows of the contingency table:

      >>> from sklearn.metrics.cluster import completeness_score
      contmtx = np.array([[8, 0, 0, 0],
                          [8, 0, 0, 0],
                          [0, 0, 8, 0]])
      >>> peakiness_metric(contmtx)
      1.0

    Even though this contingency matrix is not completely associative, it is
    still completely peaky.
         
    """
    
    
    return associativity_peakiness_AP_score(contmtx)[1]


def AP_metric(contmtx):
    
    """AP_metric of a contingency table given a ground truth.

    The AP_metric is the harmonic mean between associativity and peakiness:

        AP_metric = 2 * (peakiness_metric * associativity_metric) /
            (peakiness_metric + associativity_metric)

    Read more in the :ref:`User Guide <associativity-peakiness>`.

    Parameters
    ----------
    contmtx : array-like of shape (n_truth_classes, n_clusters)
        Ground truth class labels are the rows, and cluster indices are the columns.

    Returns
    -------
    AP_metric : float
       Score between 0.0 and 1.0. 1.0 stands for perfectly associative and peaky labeling.

    See Also
    --------
    associativity_metric : Associativity between truth labels and clusters in a 
        contingency table as determined from the peak value in each row of the 
        contingency matrix.
    peakiness_metric : Peakiness of the peak values in rows of a contingency 
        table. The peakiness metric meaures the credence of the associativity score.
    AP_metric : AP metric, the harmonic mean of associativity and peakiness metric scores.

     References
     ----------

     .. [1] `Naomi E Zirkind and William J Diehl, 2026. Associativity-Peakiness 
             Metric for Contingency Tables
        <https://doi.org/10.48550/arXiv.2604.22655f>`_


    Examples
    --------
    Perfect labelings are both associative and peaky, hence have score 1.0::

      >>> from sklearn.metrics.cluster import AP_metric
      >>> contmtx = np.array([[0, 0, 8, 0],
                              [0, 8, 0, 0],
                              [0, 0, 0, 8]])
      >>> AP_metric(contmtx)
      1.0
   

    Contingency matrices which have almost all elements in one cluster have low
    associativity, and hence give low AP score:
     >>>   contmtx = np.array([[7, 1, 0, 0],
                               [8, 0, 0, 0],
                               [6, 1, 0, 1]])
        
      
     >>> associativity_peakiness_AP_score(contmtx)
      (0.0, 0.897, 0.0) where the elements are (associativity, peakiness, AP score)

    Contingency matrices that have nearly equal numbers of elements in multiple 
    clusters give low peakiness score, and hence low AP score:
     >>>   contmtx = np.array([[3, 3, 1, 1],
                               [1, 3, 3, 1],
                               [3, 1, 1, 3]])
      
     >>> associativity_peakiness_AP_score(contmtx)
     (0.667, 0.0, 0.0)
     
    
     
    If classes members are completely split across different clusters,
    then both the associativity and peakiness are 0, hence the AP metric is null:
    contmtx = np.array([[2, 2, 2, 2],
                        [2, 2, 2, 2],
                        [2, 2, 2, 2]])
     
    >>> associativity_peakiness_AP_score(contmtx)
    (0.0, 0.0, 0.0)
     
    """
    
    return associativity_peakiness_AP_score(contmtx)[2]




      