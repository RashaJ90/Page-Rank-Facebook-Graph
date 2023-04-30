class Community(object):
    """
        Useful code for plotting communities according to their colors
    """

    def _init_(self, G):
        self.g = G

    def set_node_community(self, communities):
        """
          Add community to node attributes
        """
        for c, v_c in enumerate(communities):
            for v in v_c:
                # Add 1 to save 0 for external edges
                self.g.nodes[v]['community'] = c + 1

    def set_edge_community(self):
        """
          Find internal edges and add their community to their attributes
      """
        for v, w, in self.g.edges:
            if self.g.nodes[v]['community'] == self.g.nodes[w]['community']:
                # Internal edge, mark with community
                self.g.edges[v, w]['community'] = self.g.nodes[v]['community']
            else:
                # External edge, mark as 0
                self.g.edges[v, w]['community'] = 0

    def get_color(self, i, r_off=1, g_off=1, b_off=1):
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
