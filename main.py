from random import sample
import community as community_louvain
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nxcom
from page_rank import PageRank, PersonalizedPageRank  # An algorithm for community detection
from visualize import display_graph, plot_degree_coef, plot_graph, communities_graph, plot_page_rank_indegree, plot_page_rank_pr_node


def Erdos_Renyi_Graph(n, p):
    """
        Erdos-Renyi random graph (a.k.a. Poisson Metwork) is a random graph
    G(N, p) with N labeled nodes where each pair of nodes is connected by
    a preset probability p.
     simple models of networks like this can give us a feel for
    how more complicated real-world systems should behave in general
    """
    G = nx.erdos_renyi_graph(n, p, seed=123)
    return G


def main():
    # Load facebook social network data
    G_social = nx.read_edgelist(
        r"data/facebook_combined.txt")
    G_social = nx.relabel_nodes(G_social, {str(i): i for i in range(
        G_social.number_of_nodes())})  # change node names to integers (recommended)

    print(nx.info(G_social))
    display_graph(G_social)

    """    
        For a graph  𝐺=(𝑉,𝐸), we define the local clustering coefficient of vertex  𝑉𝑖 as  𝐶𝑖=2|Δ𝑖|𝑑𝑖(𝑑𝑖+1) , where:
        𝑑𝑖 is the degree of the vertex  𝑉𝑖.
        Δ𝑖={(𝑗,𝑘)𝑠.𝑡.𝑒𝑖𝑗,𝑒𝑗𝑘,𝑒𝑖𝑘∈𝐸} is the set of closed triangles containing  𝑉𝑖.
        𝐶𝑖  measures how likely are two neighbours of vertex  𝑉𝑖 to have a direct edge connecting between them:
        it is a measure of the degree to which nodes in a graph tend to cluster together!
    """

    d = dict()
    for n in G_social.nodes():
        d[nx.clustering(G_social, n)] = nx.degree(G_social, n)
    df = pd.DataFrame(d.items(), columns=['clustercoef', 'degree'])
    print(df.corr(method='pearson'))
    # From the graph we can see how vertices with higher degree have a lower clustering coefficients![low correlation]
    plot_degree_coef(df)

    # Average of local clustering coefficients:
    avg_cluster = nx.average_clustering(G_social)
    print("The average of the local clustering coefficients over all nodes for the facebook graph is:", avg_cluster)

    n = G_social.number_of_nodes()
    E = G_social.number_of_edges()
    p = E / (n * (n - 1) / 2)  # 2*np.log(n_nodes)/n_nodes
    G_Erdos_Renyi = Erdos_Renyi_Graph(n, p)
    print("The average of the local clustering coefficients over all nodes for the Erdos-Renyi random graph is:",
          nx.average_clustering(G_Erdos_Renyi))

    """
        The average of the local clustering coefficients of Erdos-Renyi random graph is less than the facebook graph
        where its nodes tend to create tight knitt groups characterized by a relatively high density of ties;
        this likelihood tends to be greater than the average probability of a tie randomly established between two nodes (Erdos-Renyi random graph)
    """
    # Page-Rank:
    pr = PageRank(G_social, 0.85, 0.0001)
    plot_page_rank_indegree(G_social, pr)
    plot_page_rank_pr_node(n, pr)

    # Personalized Page-Rank:
    ppr_allnodes = PersonalizedPageRank(G_social, G_social.nodes())
    ppr1_200, _ = zip(*sorted(ppr_allnodes.items(), key=lambda x: x[1], reverse=True)[:200])
    plot_graph(G_social, ppr1_200)

    random_node = sample(G_social.nodes(), 1)
    ppr_onenode = PersonalizedPageRank(G_social, random_node)  # [4023]
    ppr2_200, _ = zip(*sorted(ppr_onenode.items(), key=lambda x: x[1], reverse=True)[:200])
    plot_graph(G_social, ppr2_200)
    """
         High pageranked nodes are established all over the facebook graph, which means that 
         they are centered in highest areas of linkeage(the highest is in the center of the graph),
          while in (b) they are clustered in the area around the node we chose with a difference of nodes pagerankings.
    """

    # Using communities structures:
    communities = nxcom.greedy_modularity_communities(G_social)
    M1 = nxcom.modularity(G_social, communities)
    # Count the communities
    print(
        f"The Facebook Social Network has {len(communities)} communities(greedy Alg.) With Modularity of {round(M1, 2)}")
    communities_graph(G_social, sorted(communities, key=len, reverse=True))

    # Using communities structures with Louvain algorithm:
    partition = community_louvain.best_partition(G_social)
    M2 = community_louvain.modularity(partition, G_social)
    # Count the communities
    print(
        f"The Facebook Social Network has {len(set(partition.values()))} communities(louvain Alg.) With Modularity of {round(M2, 2)}")
    louvain = []
    for i in list(set(partition.values())):
        l = [k for k, v in partition.items() if i == v]
        louvain.append(frozenset(l))

    communities_graph(G_social, sorted(louvain, key=len, reverse=True))
    """
        When partitioning a graph into communities, we can define a quantity called modularity that 
        measures how well is the graph separated into different communities. 
        Communities found 13 clusters, while louvain found 14 with a higher modularity of 83%. 
        The difference we can see between the two graphs is that louvain clusters are distributed 
        closely to each other (we can see that by the transparency of the clouds colors)
    """


if __name__ == "__main__":
    main()
