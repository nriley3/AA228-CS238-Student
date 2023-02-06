import sys

import networkx as nx
import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def bayes_score_component(M,alpha):
    # print("Alpha: ", alpha)
    # print("M: ", M)
    p = np.sum(sc.special.loggamma(alpha+M))
    p -= np.sum(sc.special.loggamma(alpha))
    p += np.sum(sc.special.loggamma(np.sum(alpha,axis=1)))
    p -= np.sum(sc.special.loggamma(np.sum(alpha,axis=1)+np.sum(M,axis=1)))
    return p

def bayes_score(vars,G,D):
    n = len(vars)
    M = statistics(vars,G,D)
    alpha = prior(vars,G,D)
    return np.sum([bayes_score_component(M[i],alpha[i]) for i in np.arange(n)])

def statistics(vars,G,D): # Note to self: This works!!!
    n = D.shape[1] # how many variables there are
    # r = [D[vars[i]].max() for i in np.arange(n)] # how many values in each variable, according to max value, only use if panda
    r = [D[:,i].max() for i in np.arange(n)] # same thing as above, but if D is a np array
    q = [int(np.prod([r[j] for j in G.predecessors(i)])) for i in np.arange(n)] # parental instantiations
    M = [np.zeros([q[i],r[i]]) for i in np.arange(n)]
    # print("M init: ", M)
    for row in np.arange(D.shape[0]): # iterate over each row in D
        # o = D.iloc[row] # use if D is panda
        o = D[row] # use if D is np array
        # print("o: ", o)
        for i in np.arange(n):
            k = o[i]
            parents = [index for index in G.predecessors(i)]
            j = 0
            if not (parents == []): # if not empty (ie, if there are parents)
                # print("NOT EMPTY!!!")
                o_parents = tuple([o[index]-1 for index in parents]) # values of parents (coord)
                r_parents = tuple([r[index] for index in parents]) # number of instantiations of parents (size)
                # print("o_parents: ", o_parents)
                # print("r_parents",r_parents)
                j = np.ravel_multi_index(o_parents,r_parents)
                # print("j: ",j)
            M[i][j,k-1] += 1
    return M

def sub2ind(siz,x):
    print("x: ", (x))
    print("siz: ", siz[:-1])
    k = np.vstack(([1],[np.cumprod(siz[:-1])]))
    j = np.dot(k,x-1)+1
    return j

def prior(vars,G,D): # This works!!
    n = len(vars)
    # r = [D[vars[i]].max() for i in np.arange(n)] # use if D is panda
    r = [D[:,i].max() for i in np.arange(n)] # use if D is np array
    q = [int(np.prod([r[j] for j in G.predecessors(i)])) for i in np.arange(n)]
    return [np.ones([q[i],r[i]]) for i in np.arange(n)]

class K2Search:
    def __init__(self,ordering):
        self.ordering = ordering # ordering set when initialized

def fit(method,vars,D):
    G = nx.DiGraph()
    G.add_nodes_from(method.ordering) # starts graph with a node for every variable, ie nodes 0 to n-1
    # for (k,i) in enumerate(method.ordering[1:]):
    for k in method.ordering[1:]:
        print("k: ", k)
        i = k
        n_parents = 0
        # print("For this loop: ")
        # print("k: ", k)
        # print("i: ", i)
        y = bayes_score(vars,G,D) # Bayes score of current graph
        while True:
            y_best,j_best = -np.inf,0
            for j in method.ordering[0:k-1]:
                if not G.has_edge(j,i): # if this edge doesn't exist
                    G.add_edge(j,i)
                    yprime = bayes_score(vars,G,D) # Bayes score of graph with added edge
                    # print("Bscore: ", yprime)
                    if yprime > y_best: #and nx.is_directed_acyclic_graph(G): # If added edge is the best
                        y_best,j_best = yprime,j # Keep track of it
                    G.remove_edge(j,i) # Remove so we can test the next edge
            if y_best > y: # If the best edge was better than before
                y = y_best # Update
                G.add_edge(j_best,i) # Add the best edge
                n_parents += 1
            else: # If best edge was worse
                break # Break out of loop for this k
            if n_parents > 10: # if there are 11 or more parents
                break
    print("Bscore: ", y)
    return G



def compute(infile, outfile):
    start = time.time()
    # G = nx.DiGraph()
    # vars = [0,1,2]

    # G.add_node(0)
    # G.add_node(1)
    # G.add_node(2)

    # G.add_edge(0,2)
    # G.add_edge(1,2)

    # D = np.array([ [1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1],
    #                 [1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1],
    #                 [3, 1, 1, 2, 3, 3, 2, 1, 3, 2, 1]
    #             ]).T
    # print("D: ", D)

    # M = statistics(vars, G, D)
    # alpha = prior(vars, G,D)
    # y = bayes_score(vars, G, D)

    # print("score: ", y)

    # # End of test! Yay!

    D = pd.read_csv(infile)
    vars = list(D)
    D = np.array(D)
    # print("D: ", D)
    # print("vars: ",vars)
    method = K2Search(np.arange(len(vars)))
    # print("Ordering: ", method.ordering)
    G = fit(method,vars,D)
    varlist = dict(zip(method.ordering,vars))
    # print("List: ", varlist)
    print("G: ", G)
    write_gph(G,varlist,outfile)
    nx.draw(G, labels=varlist,with_labels=True, font_weight='bold',pos=nx.circular_layout(G))
    plt.savefig(outfile[:-4] + ".png")
    total_time = time.time() - start
    print("Total time: ", total_time)



    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    pass


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
