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


