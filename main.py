import csv
from operator import itemgetter
import networkx as nx  # networkx - python library for dealing with graphs
from networkx.algorithms import \
    community  # This part of networkx, for community detection, needs to be imported separately.
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nxcom
import page_rank as community_louvain  # An algorithm for community detection


# %matplotlib inline

def Erdos_Renyi_Graph(n,p):
    G = nx.erdos_renyi_graph(n, p, seed=123)
    print("The average of the local clustering coefficients over all nodes for the Erdos-Renyi random graph is:",
          nx.average_clustering(G))


def main():
    # Load facebook social network data
    G_social = nx.read_edgelist(
        r"data/facebook_combined.txt")
    G_social = nx.relabel_nodes(G_social, {str(i): i for i in range(
        G_social.number_of_nodes())})  # change node names to integers (recommended)

    print(nx.info(G_social))

    # For a graph  𝐺=(𝑉,𝐸), we define the local clustering coefficient of vertex  𝑉𝑖 as  𝐶𝑖=2|Δ𝑖|𝑑𝑖(𝑑𝑖+1) , where:
    # 𝑑𝑖 is the degree of the vertex  𝑉𝑖.
    # Δ𝑖={(𝑗,𝑘)𝑠.𝑡.𝑒𝑖𝑗,𝑒𝑗𝑘,𝑒𝑖𝑘∈𝐸} is the set of closed triangles containing  𝑉𝑖.
    # 𝐶𝑖  measures how likely are two neighbours of vertex  𝑉𝑖 to have a direct edge connecting between them.
    n = G_social.number_of_nodes()  # number of nodes of facebook graph
    E = G_social.number_of_edges()  # number of edges of facebook graph
    p = E / (n * (n - 1) / 2)  # 2*np.log(n_nodes)/n_nodes
    Erdos_Renyi_Graph(n, p)


if __name__ == "__main__":
    main()
