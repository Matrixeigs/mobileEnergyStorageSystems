import numpy as np
from numpy import array, zeros, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import cplex as cpx

class TransportationSystem:
    '''

    '''

    def __init__(self, name, tsc, edge_factor=1):
        '''
        Initialization of transportation system instance
        :param name:
        :param tsc: transportation system case, in ndarray type.
        '''
        import networkx as nx

        # instance's name
        self.name = name

        # form DataFrame from ndarray
        self.node = DataFrame(tsc['node'])
        self.n_node = self.node.shape[0]

        self.edge = DataFrame(tsc['edge'])
        self.n_edge = self.node.shape[0]

        self.edge['length'] *= edge_factor

        # generate original graph for transportation network, which is frozen
        # and cannot be modified, existing node and edge data can still be redefiend.
        self.original_graph = nx.freeze(nx.from_pandas_edgelist(df=self.edge,
            source='init_node', target='term_node', edge_attr='length'))

        # Incrementally updated graph for transportation network
        self.update_graph()

        pass

    def update_graph(self, off_road=[]):
        """
        off_road: list of outage roads at this point.
        edge_factor: scale up or down edge length
        generate and update transportation system graph.
        :return:
        """
        import networkx as nx

        # Get the graph before information updating
        self.graph = nx.Graph(self.original_graph)
        self.graph.remove_edges_from(off_road)

        return self.graph


if __name__ == "__main__":

    pass


