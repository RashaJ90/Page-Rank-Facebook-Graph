from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from communites import Community

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})


def get_color(i, r_off=1, g_off=1, b_off=1):
    """
        Assign a color to a vertex.
    """
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)


def display_graph(graph):
    pos = nx.spring_layout(graph)
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (20, 15)
    nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=False, edge_cmap='YlGnBu', node_size=35)
    plt.axis('off')
    plt.show()


def plot_degree_coef(df):
    colors = list("rgbcmyk")
    plt.scatter(df['clustercoef'], df['degree'], color=colors.pop())
    plt.title("Local clustering coefficient for all nodes of the facebook graph")
    plt.xlabel("clustering coefficients")
    plt.ylabel("degrees")
    plt.show()


def plot_page_rank_pr_node(n, pr):
    # Plotting the page rank score
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(np.arange(n), np.sort(pr))
    ax.set_ylabel('Page-Rank')
    ax.set_xlabel('Node ID')
    plt.show()


def plot_page_rank_indegree(G, pr):
    # Plot page rank vs. in-degree
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(np.fromiter(G.number_in_links.values(), dtype=float), pr, 'o', color='black')
    ax.set_ylabel('Page-Rank')
    ax.set_xlabel('In-Degree')
    plt.show()


def plot_graph(G, ppr):
    D = G.subgraph([i for i in list(G.nodes()) if i not in ppr])
    H = G.subgraph(ppr)
    pos = nx.spring_layout(G)  # setting layout
    betCent1 = nx.betweenness_centrality(D, normalized=True, endpoints=True)
    betCent2 = nx.betweenness_centrality(H, normalized=True, endpoints=True)
    betCent = {**betCent1, **betCent2}
    betCent = {k: v for k, v in sorted(betCent.items(), key=lambda x: x[0], reverse=False)}

    node_size = [v * 10000 for v in betCent.values()]
    # node_color = [get_color(v) for v in  ppr]
    node_color = []
    for v in G.nodes():
        if v in ppr:
            node_color.append(get_color(v))
        else:
            node_color.append((0.4117647058823529, 0.4117647058823529, 0.4117647058823529))

    # node_size2 =  [v * 10000  for v in ppr.values()]
    plt.rcParams['figure.figsize'] = (20, 15)

    nx.draw_networkx(G, pos=pos, with_labels=False,
                     node_color=node_color, edge_color="darkgray",
                     node_size=node_size)
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    # node_color="dimgray",edge_color="darkgray",
    # node_size=node_size1)
    plt.axis('off')
    plt.title("Facebook Social Network data-Nodes colored by their PageRanks(Best 200 nodes)")
    plt.show()


def communities_graph(G, communities):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize': (15, 10)})
    plt.style.use('dark_background')
    pos = nx.spring_layout(G, k=0.1)
    # Set node and edge communities
    Community.set_node_community(G, communities)
    Community.set_edge_community(G)

    # Set community color for internal edges
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ["black" for e in internal]
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
    # external edges
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=0,
        edgelist=external,
        edge_color="silver",
        node_color=node_color,
        alpha=0.2,
        with_labels=False)
    # internal edges
    nx.draw_networkx(
        G, pos=pos,
        edgelist=internal,
        edge_color=internal_color,
        node_color=node_color,
        alpha=0.05,
        with_labels=False)
    plt.show()
