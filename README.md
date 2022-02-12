# MDL-network-population-clustering

<ins>Inputs</ins> \
D: list of sets. the s-th set contains all the edges (i,j) in the s-th network in the sample (does not include the other direction (j,i)). the order of edgesets within D only matters for contiguous clustering, where we want the edgesets to be in order of the samples in time \
N: number of nodes in each network \
K0: initial number of clusters (for discontiguous clustering, usually K0 = 1 works well) \
n_fails: number of failed reassign/merge/split/merge-split moves before terminating algorithm 

<ins>Outputs</ins> \
C: dictionary with items (cluster label):(set of indices corresponding to networks in cluster). \  
E: dictionary with items (cluster label):(edge count dictionary). edge count dictionary is a dictionary with items (edge (i,j)):(number of times edge (i.j) occurs in cluster) \
A: dictionary with items (cluster label):(set of edges corresponding to mode of cluster) \
L: inverse compression ratio (description length after clustering)/(description length of naive transmission) 

<ins>For discontiguous clustering, use:</ins> \
MDLobj = MDL_populations(D,N,K0,n_fails) \
MDLobj.initialize_clusters() \
C,E,A,L = MDLobj.run_sims() 

<ins>For contiguous clustering, use:</ins> \
MDLobj = MDL_populations(D,N,K0=(anything),n_fails=(anything)) \
MDLobj.initialize_contiguous() \
C,E,A,L = MDLobj.run_sims_contiguous() 
