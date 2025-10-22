import numpy as np
import pandas as pd
import networkx as nx
import scipy
from scipy.spatial.distance import pdist, squareform
import sklearn
import matplotlib.pyplot as plt
from collections import Counter
import random
from sklearn.model_selection import KFold
import tqdm
import requests
import time

def load_network_sparse(net_file,ngene):
    ppi_df = pd.read_csv(net_file,header=None,sep='\t')
    A = np.zeros((ngene,ngene))
    row_idx = ppi_df.iloc[:,0].values -1 
    col_idx = ppi_df.iloc[:,1].values -1
    A[row_idx, col_idx] = ppi_df.iloc[:,2].values
    assert (A == A.T).all()
    zero_rows = np.all(A == 0, axis=1)
    diag_indices = np.arange(ngene)
    A[diag_indices[zero_rows], diag_indices[zero_rows]] = 1
    return A


def load_all_nets(ppi_files,n_gene):
    '''
    parameters:
    - ppi_files: [str, str, ...], list of network file paths, each file should contain three columns: [protein1, protein2, score]
    - ref_gene_file: str, file path, the file contains all genes, one gene per line
    output:
    - nets: n_file x n_gene x n_gene array with ppi networks
    '''
    n_file = len(ppi_files)
    nets = np.zeros((n_file,n_gene,n_gene))
    for i in range(n_file):
        A = load_network_sparse(ppi_files[i],n_gene)
        nets[i,:,:] = A
    return nets


def compute_rwr_original_sparse(ppi_files,restart_prob,ngene,nets):
    ''' 
    - ppi_files: list of network file paths
    - restart_prob: RWR restart probability
    - ngene: number of genes

    walks[i,:,:]: each column is the stationary distribution of a node
    '''
    n_file = len(ppi_files)
    e = np.ones(ngene)
    I = np.eye(ngene)
    walks = np.zeros((n_file,ngene,ngene))
    for i in range(n_file):
        A = nets[i,:,:]
        d = A @ e
        P = A / d # transition matrix
        W = (I - (1 - restart_prob) * P)
        W = np.linalg.inv(W)
        W = W * restart_prob 
        walks[i,:,:] = W
    return walks


# there are subtle differences between matlab implementation and python implementation.
# It's caused by after RWR, there are some values that are extremely small, 
# for example, in network 6, [1825,943], its value is 1e-17 ish, after taking log, the log becomes -38
# these differences accumulated and as a result, the eigenvalues become different.
# After filtering out the extreme numbers in matlab code, the two results become the same.
def svd_embed_sparse_func(walks, ngene, embed_dim):
    n_net = walks.shape[0]
    mat = np.zeros((ngene,ngene))
    W_updated = np.zeros_like(walks)
    for i in range(n_net):
        W = walks[i,:,:]
        W[W<=1e-8] = 0
        W = np.log(W, where = W > 1e-8)
        W_updated[i,:,:] = W
        tmp = W.T @ W
        mat = mat + tmp
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(mat,k=embed_dim)
    x = np.diag(np.sqrt(np.sqrt(eigenvalues))) @ eigenvectors.T
    return np.real(x)

def load_train_test_anno(rand,fold,org,ont_type,ont_size1,ont_size2):
    '''
    predifined fold splits
    - rand: 1 2 3 4 5
    - fold: 1 2 3 4 5
    - org: "Ecoli" or "yeast"
    '''
    file_name = 'data/train_test_split/'+org+'/rand' + str(rand) +'/fold' + str(fold) + '_' + ont_type+ '_' +  str(ont_size1)+ '_' +  str(ont_size2)+ '_train_anno.txt'
    train = pd.read_csv(file_name,header=None,sep = '\t')
    file_name = 'data/train_test_split/'+org+'/rand' + str(rand) +'/fold' + str(fold) + '_' + ont_type+ '_' +  str(ont_size1)+ '_' +  str(ont_size2)+ '_test_anno.txt'
    test = pd.read_csv(file_name,header=None,sep = '\t')
    return train.to_numpy(), test.to_numpy()


def augment_graph(nets, ngene, gene_clusters, mustlink_weight, cannotlink_weight):
    '''
    - nets: original adjacency matrices directly read from PPI files
    - gene_clusters: (num_clusers, num_genes), binary matrix indicating which gene belongs to which clusters
    '''
    n_nets = nets.shape[0]
    n_clusters = gene_clusters.shape[0]
    augmented = np.zeros((n_nets,(ngene+n_clusters),(ngene+n_clusters)))
    for i in range(n_nets):
        A = nets[i,:,:]
        A_block = np.block([[A,mustlink_weight*gene_clusters.T],[mustlink_weight*gene_clusters,cannotlink_weight*np.ones((n_clusters,n_clusters))]])
        np.fill_diagonal(A_block,0)
        zero_rows = np.all(np.absolute(A_block) == 0, axis=1)
        diag_indices = np.arange(ngene+n_clusters)
        A_block[diag_indices[zero_rows], diag_indices[zero_rows]] = 1
        augmented[i,:,:] = A_block
    return augmented


def augmented_RWR(augmented_nets, restart_prob):
    n_nets = augmented_nets.shape[0]
    n_nodes = augmented_nets.shape[1]
    augmented_walks = np.zeros((n_nets,n_nodes,n_nodes))
    e = np.ones(n_nodes)
    for i in range(n_nets):
        A = augmented_nets[i,:,:]
        d = np.absolute(A) @ e
        L = np.diag(d) - (1-restart_prob)*A
        L_inv = np.linalg.inv(L)
        W = restart_prob*(np.diag(d) @ L_inv)
        augmented_walks[i,:,:] = W
    return augmented_walks


def augmented_SVD_with_cannolink(aug_walks, embed_dim):
    n_net = aug_walks.shape[0]
    n_node = aug_walks.shape[1]
    mat = np.zeros((n_node,n_node))
    W_updated = np.zeros_like(aug_walks)
    for i in range(n_net):
        W = aug_walks[i,:,:]
        min_entry = W.min()
        if min_entry > 0:
            min_entry = 0.0
        W = W - min_entry 
        W[W<=1e-8] = 0
        W = np.log(W, where = W > 1e-8)
        W_updated[i,:,:] = W
        tmp = W.T @ W
        mat = mat + tmp
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(mat,k=embed_dim)
    x = np.diag(np.sqrt(np.sqrt(eigenvalues))) @ eigenvectors.T
    return np.real(x)


def get_knn_ind(embed,train_anno):
    '''
    parameters:
    - embed: (dim, num_gene), protein embeddings
    - train_anno: annotations for training proteins
    output:
    dist_mat: n_gene x n_gene
    sorted_ind: ngene x (ngene-1), top n labels
    '''
    n_gene = train_anno.shape[1]
    train_idx = np.where(sum(train_anno)>0)[0]
    dist_mat = squareform(pdist(embed.T))
    dist_mat = dist_mat[:n_gene,:n_gene] # symmetrical
    np.fill_diagonal(dist_mat, 1e8)
    sorted_ind = np.argsort(dist_mat, axis=1)
    
    return dist_mat, sorted_ind


def majority_vote(dist_mat, knn_mat, train_anno, test_anno,k, weighted=True):
    '''
    parameters:
    - dist_mat: n_gene x n_gene
    - knn_mat: ngene x (ngene-1), sorted labels
    - train_anno: n_label x n_gene
    - test_anno: n_label x n_gene
    - k: number of nearest neighbors
    - weighted: boolean, whether doing weighted majority vote or not
    output:
    - final_scores: n_label x n_test, normalized scores of each label
    - num_voters: vector of numbers of voting nodes
    '''
    train_idx = np.where(sum(train_anno)>0)[0]
    test_idx = np.where(sum(test_anno)>0)[0]
    final_scores = np.zeros((train_anno.shape[0],len(test_idx)))
    num_voters = []
    updated_voters = []
    c = 0
    for index, i in enumerate(test_idx):
        nn = knn_mat[i,:k]
        nn_labeled = nn[np.isin(nn, train_idx)] 
        if len(nn_labeled) == 0: # if within the first k neighbors, no neighbor is labeled, then use the nearest neighbor with label
            voting_node = knn_mat[i,:][np.isin(knn_mat[i,:], train_idx)][0]
            scores = np.array(train_anno[:,voting_node])
            scores = scores / sum(scores)
            num_voters.append(len(nn_labeled))
            tmp = [voting_node]
            updated_voters.append(tmp)
        else:
            votes = np.array(train_anno[:,nn_labeled])
            if weighted:
                d = dist_mat[i,nn_labeled]
                d = d[np.nonzero(d)]
                votes = np.array(train_anno[:,nn_labeled[np.nonzero(d)]])
                tmp = nn_labeled[np.nonzero(d)]
                updated_voters.append(tmp)
                num_voters.append(len(d))
                if len(d) == 0:
                    c += 1
                    voting_node = np.random.choice(train_idx)
                    scores = np.array(train_anno[:,voting_node])
                    scores = scores / sum(scores)
                else:
                    weights = 1 / d
                    scores = votes @ weights.T
                    scores = scores / sum(scores)
            else:
                num_voters.append(len(nn_labeled))
                updated_voters.append(nn_labeled)
                scores = np.sum(votes,axis=1)
                scores = scores / sum(scores)
        
        final_scores[:,index] = np.squeeze(scores)
    # print(c)
    return final_scores, num_voters,updated_voters


def acc_top1_pred(test_scores, test_anno):
    '''
    for each test gene, find the label with the highest predicted score, use it as the predicted label
    accuracy is defined as (#predicted label in test true labels) / (#test genes)
    problems: if there's a tie, the one with smaller index will be used
    parameters:
    - test_scores: n_label x n_test
    - test_anno: n_label x n_gene
    output:
    - acc: accuracy score
    '''
    test_idx = np.where(sum(test_anno)>0)[0]
    zero_idx = np.where(np.sum(test_scores,axis=0)==0)[0]
    mask = np.ones(len(test_idx), dtype=bool)
    mask[zero_idx] = False
    test_anno = test_anno[:,test_idx] # n_label x n_test
    sorted_index = np.argsort(-1*test_scores,axis=0) # n_label x n_test, with row 0 the highest predicted label for each gene
    true_pred = test_anno[sorted_index[0,:], np.arange(test_anno.shape[1])]
    true_pred = true_pred[mask]
    acc = np.mean(true_pred)
    return acc,true_pred
    

def f1_auprc_pred(test_scores, test_anno,top_n):
    '''
    for each test gene, find the labels with the top_n highest predicted scores, use them as the predicted labels
    f1 is defined as 2*TP / 2*TP + FP + FN
    probelms: if there's a tie, the one with smaller index will be used, only top n predictions will be considered, it will increase the number of FN
    parameters:
    - test_scores: n_label x n_test
    - test_anno: n_label x n_gene
    - top_n: int, the number of labels to be predicted
    output:
    - acc: accuracy score
    '''
    test_idx = np.where(sum(test_anno)>0)[0]
    zero_idx = np.where(np.sum(test_scores,axis=0)==0)[0]

    mask = np.ones(len(test_idx), dtype=bool)
    mask[zero_idx] = False
    test_anno = test_anno[:,test_idx] # n_label x n_test
    
    test_anno = test_anno[:,mask]
    test_scores = test_scores[:,mask]
    sorted_index = np.argsort(-1*test_scores,axis=0) # n_label x n_test, with row 0 the highest predicted label for each gene
    top_ind = sorted_index[:top_n,:].flatten()
    pred = np.zeros_like(test_anno)
    cols = np.tile(np.arange(test_anno.shape[1]), top_n)
    pred[top_ind, cols] = 1
    f1 = sklearn.metrics.f1_score(test_anno.flatten(),pred.flatten())
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(test_anno.flatten(), pred.flatten())
    auprc = sklearn.metrics.auc(recall, precision)
    return f1, auprc
    
'''
for each augmented node, randomly choose a fixed number of nodes to connect to
the number of genes that each augmented node connects to are the same except for the last one
'''
def random_split_vector(train_anno,n_gene, num_sub_vectors,seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    input_vector = np.where(sum(train_anno)>0)[0]

    if num_sub_vectors <= 0 or num_sub_vectors > len(input_vector):
        raise ValueError("Invalid number of sub-vectors")
    
    shuffled_vector = np.random.permutation(input_vector)
    sub_vector_size = len(shuffled_vector) // num_sub_vectors
    
    group_matrix = np.zeros((num_sub_vectors, len(input_vector)), dtype=int)
    res_matrix = np.zeros((num_sub_vectors, n_gene), dtype=int)
    
    start_index = 0
    for i in range(num_sub_vectors):
        end_index = start_index + sub_vector_size
        
        if i == num_sub_vectors - 1:
            end_index = len(shuffled_vector)
        
        selected_indices = shuffled_vector[start_index:end_index]
        
        group_matrix[i, np.isin(shuffled_vector, selected_indices)] = 1
        start_index = end_index
    
    res_matrix[:,shuffled_vector] = group_matrix
    
    return res_matrix


def run_pipeline(ppi_files,n_gene,method=None,restart_prob=None,embed_dim=None,rand=None,org=None,n_fold=None,k=None,ont_type=None,ont_size1=None,ont_size2=None,n_cluster=None):
    ''' 
    parameters:
    - ppi_files: list of str, list of file paths to ppi networks
    - n_gene: int, number of genes
    - method: list of str, one or more of Mashup, REPEL
    - restart_prob: float, RWR restart probability
    - embed_dim: int, number of dimension
    - rand: int, random split
    - org: str, "yeast" or "Ecoli" 
    - n_fold: int, total number of folds
    - k: int, number of nearest neighbors to be considered
    - ont_type: str, bp or mf or cc
    - ont_size1: int, 11, 31, 101
    - ont_size2: int, 30, 100, 300
    - n_cluster: int, number of random augmented nodes
    output:
    - performance_dict: a dictionary contains list of performances for all methods
    '''

    performance_dict = {}

    for m in method:
        m_acc = m + "_acc"
        m_f1 = m + "_f1"
        m_auprc = m + "_auprc"
        performance_dict[m_acc] = []
        performance_dict[m_f1] = []
        performance_dict[m_auprc] = []
        for i in range(n_fold):
            print("fold: ", i+1)
            train_anno, test_anno = load_train_test_anno(rand,i+1,org,ont_type,ont_size1,ont_size2)
            if m == "Mashup":
                print("Mashup")
                nets = load_all_nets(ppi_files,n_gene)
                walks = compute_rwr_original_sparse(ppi_files,restart_prob,n_gene,nets)
                x = svd_embed_sparse_func(walks, n_gene, embed_dim)
                dist_mat, knn = get_knn_ind(x,train_anno)
                scores, _, _ = majority_vote(dist_mat, knn, train_anno, test_anno,k)
                acc,_ = acc_top1_pred(scores, test_anno)
                f1, auprc = f1_auprc_pred(scores, test_anno,3)
                performance_dict[m_acc].append(acc)
                performance_dict[m_f1].append(f1)
                performance_dict[m_auprc].append(auprc)
            elif m == "REPEL":
                print("REPEL")
                nets = load_all_nets(ppi_files,n_gene)
                rand_cluster = random_split_vector(train_anno,n_gene, n_cluster,seed=None)
                rand_graph = augment_graph(nets, n_gene, rand_cluster, 1, -1)
                rand_rwr_res = augmented_RWR(rand_graph, restart_prob)
                mat_rand_x = augmented_SVD_with_cannolink(rand_rwr_res, embed_dim)
                dist_mat, knn = get_knn_ind(mat_rand_x,train_anno)
                scores, _,_ = majority_vote(dist_mat, knn, train_anno, test_anno,k, weighted=True)
                acc,_ = acc_top1_pred(scores, test_anno)
                f1, auprc = f1_auprc_pred(scores, test_anno,3)
                performance_dict[m_acc].append(acc)
                performance_dict[m_f1].append(f1)
                performance_dict[m_auprc].append(auprc)

            else:
                print("Haven't implemented yet")
                return
    return performance_dict


def write_log(performance_dict,rand,org,ont_type,ont_size1,ont_size2,save_path=None):
    with open(save_path,"a") as f:
        tmp = org + " " + "rand " + str(rand) + " " + ont_type + " " + str(ont_size1) + " " + str(ont_size2) 
        f.write(tmp)
        f.write("\n")
        for k, v in performance_dict.items():
            f.write(k)
            f.write(" ")
            for i in v:
                f.write(f"{i:.4f}")
                f.write(" ")
            f.write("\n")


def generate_label_matrix(communities,n_node):
    label_matrix = np.zeros((len(communities),n_node))
    for idx, comm in enumerate(communities):
        node_id = list(comm)
        label_matrix[idx,node_id] = 1
    return label_matrix


def add_edges_biased(G: nx.Graph,subset,p_intra = None,p_inter = None):
    '''
    Add edges to G inplace so that nodes inside subset get more internal edges and less external edges.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Graph that already contains all nodes. (Existing edges stay.)
    subset : iterable
        Collection of node labels you want to knit together.
    p_intra : float
        Probability of adding an edge between 2 subset nodes.
    p_inter : float
        Probability of adding an edge between a subset node and any node outside the subset.
    seed : int | None
        Optional RNG seed for reproducibility.
    '''
    
    subset = set(subset)
    all_nodes = list(G.nodes)

    for u in subset:
        for v in subset: # within cluster
            if u < v and not G.has_edge(u, v):
                if random.random() <= p_intra:
                    G.add_edge(u, v)

        for v in all_nodes: # outside of cluster
            if v not in subset and not G.has_edge(u, v):
                if random.random() <= p_inter:
                    G.add_edge(u, v)


def mask_symmetric_ones(A, frac=None, seed=None, keep_diagonal=True):
    '''
    randomly remove some edges, return the symetrical adjacency matrix

    Parameters
    ----------
    A : numpy ndarray
        Adjacency matrix of the original graph.
    frac : float
        Percentage of edges to be removed.
    seed : int | None
        Optional RNG seed for reproducibility.
    keep_diagonal : bool 
        Default True, the diagonal values will be kept. Otherwise repalce to all 0.

    '''
    rng = np.random.default_rng(seed)
    r, c = np.triu_indices_from(A, k=1) # upper triangle
    ones_mask = A[r, c] == 1 
    edge_pos = np.where(ones_mask)[0] 

    num_edge_remove = int(np.round(frac * edge_pos.size))
    if num_edge_remove == 0:
        return A.copy()

    chosen = rng.choice(edge_pos, size=num_edge_remove, replace=False)

    A_masked = A.copy()
    rows, cols = r[chosen], c[chosen]
    A_masked[rows, cols] = 0
    A_masked[cols, rows] = 0

    # if not keep_diagonal:
    np.fill_diagonal(A_masked, 0)

    zero_rows = np.all(A_masked == 0, axis=1)
    diag_indices = np.arange(A_masked.shape[0])
    A_masked[diag_indices[zero_rows], diag_indices[zero_rows]] = 1

    return A_masked


def rewire_graph(ori_A,communities,percentage,p_intra=None, p_inter=None, p_disconnect=None):
    '''
    Based on the base graph and known community information, randomly generate one sparse noisy graph

    Parameters
    ----------
    ori_A : numpy ndarray
        Adjacency matrix of the base graph.
    communities : iterable
        Collection of lists of node ids.
    percentage : float
        Percentage of nodes per community you want to remain closely clustered
    p_intra : float
        Probability of adding an edge between 2 subset nodes.
    p_inter : float
        Probability of adding an edge between a subset node and any node outside the subset.
    p_disconnect: float (optional) UPDATE: removed
        Probability of adding an edge between all disconnected nodes after all communities formed.
    '''

    G = nx.Graph()
    G.add_nodes_from(range(ori_A.shape[0]))
    for idx, comm in enumerate(communities):
        node_id = list(comm)
        k = int(len(node_id) * percentage) 
        sampled = random.sample(node_id, k)
        add_edges_biased(G, sampled, p_intra=p_intra, p_inter=p_inter)
    G.remove_edges_from(nx.selfloop_edges(G))
    # disconnected_nodes = list(nx.isolates(G))
    # add_edges_biased(G, disconnected_nodes, p_intra=p_disconnect, p_inter=p_disconnect)
    return G


def generate_base_graph(n,tau1,tau2,mu,avg_degree,max_degree,min_community,max_community,seed):
    while True:
        try:
            G_multi = nx.LFR_benchmark_graph(
                n=n,tau1=tau1,tau2=tau2,mu=mu,average_degree=avg_degree,max_degree=max_degree,min_community=min_community,max_community=max_community,seed=seed
            )
            break
        except Exception as e:
            print(f"Failed with error: {e}. Retrying...")

    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))
    print("Simple graph info: ",G)
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    print("number of communities: ",len(communities))
    labels = generate_label_matrix(communities,1000)
    kf = KFold(n_splits=5, shuffle=True)
    folds = []
    for train_idx, test_idx in kf.split(labels.T):
        folds.append((train_idx, test_idx))
    (largest_node, largest_degree) = max(dict(G.degree()).items(), key=lambda item: item[1])
    print("largest node id, largest node degree",(largest_node, largest_degree))
    return G, labels, folds, communities


def generate_syn_graph(G,communities, with_base, num_graph,percentage_list,p_intra_list,p_inter_list,p_disconnect_list):
    '''
    Parameters
    ----------
    G : nx.Graph
        The base graph 
    output
    ------
    graphs: list
        base graph along with generated random graphs
    '''

    A = nx.adjacency_matrix(G)
    A = A.toarray()
    if with_base:
        syn_nets = np.zeros((num_graph+1,A.shape[0],A.shape[0]))
        syn_nets[0,:,:] = A
        graphs = [G]
    else:
        syn_nets = np.zeros((num_graph,A.shape[0],A.shape[0]))
        graphs = []
    
    for i in range(num_graph):
        tmp_g = rewire_graph(A,communities,percentage_list[i],p_intra=p_intra_list[i], p_inter=p_inter_list[i],p_disconnect=p_disconnect_list[i])
        print("generated graph: ",tmp_g)
        graphs.append(tmp_g)
        tmp_A = nx.adjacency_matrix(tmp_g)
        tmp_A = tmp_A.toarray()
        diag_idx = np.where(np.sum(tmp_A,axis=0)==0)[0]
        tmp_A[diag_idx,diag_idx] = 1
        if with_base:
            syn_nets[(i+1),:,:] = tmp_A
        else:
            syn_nets[(i),:,:] = tmp_A

    return graphs, syn_nets


def run_synthetic_pipeline(syn_nets, labels, folds):
    mu_acc_res = []
    rand_acc_res = []
    syn_files = list(range(syn_nets.shape[0]))
    for i in range(len(folds)):
        print("fold: ", i+1)
        syn_train_anno = np.copy(labels)
        syn_train_anno[:,folds[i][1]] = 0
        syn_test_anno = np.copy(labels)
        syn_test_anno[:,folds[i][0]] = 0
        syn_nets_mu = syn_nets.copy()
        syn_walks = compute_rwr_original_sparse(syn_files,0.5,syn_nets.shape[1],syn_nets_mu)
        syn_x = svd_embed_sparse_func(syn_walks, syn_nets.shape[1], 400)
        syn_mu_dist_mat, syn_mu_knn = get_knn_ind(syn_x,syn_train_anno)
        syn_mu_scores, syn_mu_num_voters,syn_mu_voters = majority_vote(syn_mu_dist_mat, syn_mu_knn, syn_train_anno, syn_test_anno,10)
        syn_mu_acc,syn_mu_correct_pred_ind = acc_top1_pred(syn_mu_scores, syn_test_anno)
        mu_acc_res.append(syn_mu_acc)
        
        syn_nets_rand = syn_nets.copy()
        syn_rand_cluster = random_split_vector(syn_train_anno,syn_nets.shape[1], 15,seed=None)
        syn_rand_graph = augment_graph(syn_nets_rand, syn_nets.shape[1], syn_rand_cluster, 1, -1)
        syn_rand_rwr_res = augmented_RWR(syn_rand_graph, 0.5)
        syn_rand_x = augmented_SVD_with_cannolink(syn_rand_rwr_res, 400)
        syn_rand_dist_mat, syn_rand_knn = get_knn_ind(syn_rand_x,syn_train_anno)
        syn_rand_scores, syn_rand_num_voters,rand_voters = majority_vote(syn_rand_dist_mat, syn_rand_knn, syn_train_anno, syn_test_anno,10, weighted=True)
        syn_rand_acc, syn_rand_correct_pred_ind = acc_top1_pred(syn_rand_scores, syn_test_anno)
        rand_acc_res.append(syn_rand_acc)
    return mu_acc_res, rand_acc_res


def replicate(num_rep,param_dict1,param_dict2,with_base=True):
    ''' 
    within one replicate: one base graph, one label info and fold split, one set of augmented graphs
    across replicates: different base graphs generated using the same set of parameters, label and fold split according to its particular base graph
    '''
    mu_rep_list = []
    rand_rep_list = []
    for rep in range(num_rep):
        print("replicate number: ",rep)
        base_G, labels, folds, communities = generate_base_graph(**param_dict1)
        graphs, syn_nets = generate_syn_graph(base_G,communities,with_base,**param_dict2)
        mu_res, rand_res = run_synthetic_pipeline(syn_nets, labels, folds)
        mu_rep_list.append(mu_res)
        rand_rep_list.append(rand_res)
    return mu_rep_list, rand_rep_list


def averaged_perf(rep,mu_rep_list,rand_rep_list):
    mu_mean_list = []
    mu_std_list = []
    rand_mean_list = []
    rand_std_list = []
    for i in range(rep):
        mu_mean_list.append(np.mean(mu_rep_list[i]))
        mu_std_list.append(np.std(mu_rep_list[i]))
        rand_mean_list.append(np.mean(rand_rep_list[i]))
        rand_std_list.append(np.std(rand_rep_list[i]))
    print("Mashup averaged accuracy: ",np.mean(mu_mean_list))
    print("Mashup averaged std: ",np.mean(mu_std_list))
    print("REPEL averaged accuracy: ",np.mean(rand_mean_list))
    print("REPEL averaged std: ",np.mean(rand_std_list))

    