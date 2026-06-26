Distribution Statement A. Approved for public release: distribution is unlimited.

Associativity-Peakiness Metric for Contingency Tables

2.3.11.10 Associativity-Peakiness Metric   (Or 2.3.11.8.1 – a subset of the Contingency Matrix section)

Contingency tables are effective in summarizing the performance of a clustering algorithm in cases in which the ground truth labels are known, analogous to the role that confusion matrices play in summarizing the performance of supervised learning algorithms.

The associativity-peakiness (AP) metric characterizes the performance of clustering algorithms as presented in contingency tables. The value of the metric is derived from the entries in the contingency table. The associativity metric evaluates the extent of one-to-one correspondence between the truth classes and the clusters formed. The peakiness metric evaluates the confidence level of the data used to calculate the associativity metric. Both of these metrics are bounded below by 0.0 and above by 1.0 (higher is better): The AP metric is the harmonic mean of the associativity metric and the peakiness . 

Since the data in vector pairs of truth values and predicted values can be expressed as a contingency table, the AP metric can be applied to such vector pairs as well as to contingency tables.

Mathematical formulation:

Let the contingency table be denoted at a matrix T_(i,j)  where the rows 𝑖 are the truth classes and the columns 𝑗 are the cluster indices. 

Associativity metric:

The associativity metric quantifies the extent of one-to-one correspondence between truth classes and clusters formed. The first step is to find, for each truth class, the cluster number with the largest value, and to form an unordered list 𝐿 of these numbers as defined here:

        L= {〖argmax〗_j (T_(i,j) ) }

The list 𝐿 is then used to construct an unordered list M of all possible pairs of distinct elements of the list 𝐿, as defined here:

        M= {(L_m,L_n ),   m≠n}

The associativity metric 𝐴 is derived from the list 𝑀 as shown:

        A= (Number of pairs in M whose members are unequal)/(Total number of pairs in M)

Peakiness metric:

The Associativity metric is derived from peak values in each row of the contingency table. The Peakiness metric measures how peaky these peaks are. It gives a confidence measure for the Associativity metric. Before defining the Peakiness metric, define the following two quantities:

        〖max〗_1 (N)=largest value of list N
        〖max〗_2 (N)=second largest value of list N

For each truth value, the two largest values are found, and the peakiness value for the elements in row 𝑖 is given by

        P_i= ((〖max〗_1 (row i)- 〖max〗_2 (row i) ))/(〖max〗_1 (row i) )

Special care must be taken in the case when 〖max〗_1 (row i)=0 , which would happen only when all elements of a row are zero. In that case, the peakiness metric is undefined for that row, so that row is excluded from the calculation of the peakiness metric.

The overall peakiness metric is the mean of each of the row values given by the above equation for P_i. 

The Associativity-Peakiness metric is the harmonic mean of the Associativity metric and the Peakiness metric.

Code sample:

        from sklearn.metrics.cluster import contingency_matrix
        
        x = ["a", "a", "a", "b", "b", "b"]
        
        y = [0, 0, 1, 1, 2, 2]
        
        contmtx = contingency_matrix(x, y)
        
        contmtx
        
        array([[2, 1, 0],
              	 [0, 1, 2]], dtype=int64)
        
        AP_metric = calcAPmetric(contmtx)
        
        AP_metric
        	0.66667 

Advantages:
	AP metric operates directly on elements of contingency table.
	Values of AP metric are between 0 and 1.
	AP metric is computationally simple.
	AP metric is analogous to the mean accuracy for confusion matrices  [1].
	AP metric has higher dynamic range than several scikit-learn metrics  [1].
	Can effectively characterize contingency tables with random assignments  [1].

Drawbacks:
	Since AP metric is derived from elements of contingency table, it requires knowledge of ground truth. 

Reference: 
[1] 	N. E. Zirkind and W. J. Dishl, "Associativity-Peakiness Metric for Contingency Tables," arXiv, https://arxiv.org/abs/2604.22655v2 or https://doi.org/10.48550/arXiv.2604.22655, 2026.


