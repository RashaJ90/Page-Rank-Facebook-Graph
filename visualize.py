from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})


def plot_degree_coef(G):
    d=dict()
    colors = list("rgbcmyk")
    for n in G.nodes():
        d[nx.clustering(G,n)]=nx.degree(G,n)
    df = pd.DataFrame(d.items(), columns=['clustercoef', 'degree'])
    plt.scatter(df['clustercoef'],df['degree'],color=colors.pop())
    plt.title("Local clustering coefficient for all nodes of the facebook graph")
    plt.xlabel("clustering coefficients")
    plt.ylabel("degrees")
    plt.show()
    print(df.corr(method='pearson'))