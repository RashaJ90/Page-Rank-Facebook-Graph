import numpy as np
import random
import scipy.sparse
import networkx as nx


class web:
    """
    https://michaelnielsen.org/blog/using-your-laptop-to-compute-pagerank-for-millions-of-webpages/
    Generates a random web link structure, and finds the
    corresponding PageRank vector.  The number of inbound
    links for each page is controlled by a power law
    distribution.

    This code should work for up to a few million pages on a modest machine.
    """

    def __init__(self, n):
        self.size = n
        self.in_links = {}
        self.number_out_links = {}
        self.number_in_links = {}
        self.dangling_pages = {}
        for j in range(n):
            self.in_links[j] = []
            self.number_out_links[j] = 0
            self.number_in_links[j] = 0
            self.dangling_pages[j] = True


def paretosample(n, power=2.0):
    """
    Returns a sample from a truncated Pareto distribution
    with probability mass function p(l) proportional to
    1/l^power.  The distribution is truncated at l = n.
    """
    m = n + 1
    while m > n:
        m = np.random.zipf(power)
    return m


def random_web(n=1000, power=2.0):
    """
    Returns a web object with n pages, and where each
    page k is linked to by L_k random other pages.  The L_k
    are independent and identically distributed random
    variables with a shifted and truncated Pareto
    probability mass function p(l) proportional to
    1/(l+1)^power.
    """
    g = web(n)
    for k in range(n):
        lk = paretosample(n + 1, power) - 1
        values = random.sample(range(n), lk)
        g.in_links[k] = values
        for j in values:
            if g.number_out_links[j] == 0: g.dangling_pages.pop(j)
            g.number_out_links[j] += 1

    # Count in links
    for k in range(n):
        g.number_in_links[k] = len(g.in_links[k])

    return g


def step(g, p, beta=0.85):
    """
    Performs a single step in the PageRank computation,
    with web g and parameter s.  Applies the corresponding M
    matrix to the vector p, and returns the resulting
    vector.
    arguments:
        g - adjancency matrix representing the network (in sparse format!)
        p - vector
        beta -  non-dangling probability
    returns:
        v - vector = M*v = (beta*g + (1-beta)*D) * v
    """
    n = g.size
    v = np.matrix(np.zeros((n, 1)))
    inner_product = sum([p[j] for j in g.dangling_pages.keys()])
    for j in range(n):
        v[j] = beta * sum([p[k] / g.number_out_links[k]
                           for k in g.in_links[j]]) + beta * inner_product / n + (1 - beta) / n
    # We rescale the return vector, so it remains a
    # probability distribution even with floating point
    # roundoff.
    return v / np.sum(v)


def PageRank(g, beta=0.85, tolerance=0.00001):
    """
    Returns the PageRank vector for the web g and
    parameter s, where the criterion for convergence is that
    we stop when M^(j+1)P-M^jP has length less than
    tolerance, in l1 norm.
    arguments:
        g - adjancency matrix representing the network (in sparse format!)
        tolerance - stopping criteria
        beta -  non-dangling probability
    returns:
        p - vector of page-rank values
    """
    n = g.size
    p = np.matrix(np.ones((n, 1))) / n
    iteration = 1
    change = 2
    while change > tolerance:
        print("Iteration: ", iteration)
        new_p = step(g, p, beta)
        change = np.sum(np.abs(p - new_p))
        print("Change in l1 norm: ", change)
        p = new_p
        iteration += 1
    return np.array(p).reshape(n)  # return one dimensional array


def PersonalizedPageRank(G, U, T=100, tolerance=1e-06, beta=0.85):
    """
        Compute Personalized page-rank:
        The function then outputs the personalized page-rank vector for the graph G and given these parameters
        The vector is defined as the stationary distribution of a random walk that with probability  1−𝛽
          moves from a node to one of its neighbours, and with probability  𝛽
          moves to a random node in the set  𝑈

        Input:
        G - graph  𝐺=(𝑉,𝐸)
        U - a subset of nodes given as a list  𝑈⊂𝑉
        T - a maximal number of iterations
        tolerance - stopping criteria
        𝛽 - probability to moves to a random node
        Output:
        p - vector of Personalized page-rank values
    """
    N = len(G)
    nodelist = list(G.nodes())
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, dtype=float)
    S = np.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    M = Q * M

    # initial vector
    x = np.repeat(1.0 / N, N)

    # Personalization vector
    personalization = {}
    for v in nodelist:
        if v in U:
            personalization[v] = 1
        else:
            personalization[v] = 0
    p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
    p = p / p.sum()

    # Dangling nodes
    is_dangling = np.where(S == 0)[0]
    # power iteration: make up to max_iter iterations
    for _ in range(T):
        xlast = x
        x = beta * x * M + beta * sum(x[is_dangling]) + (1 - beta) * p

        # check convergence, l1 norm
        change = np.sum(np.abs(x - xlast))
        if change < tolerance:
            return dict(zip(nodelist, map(float, x)))
