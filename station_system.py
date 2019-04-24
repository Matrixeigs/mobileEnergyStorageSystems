import numpy as np
from numpy import array, zeros, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import cplex as cpx

class StationSystem:
    def __init__(self, name, ssc):
        """
        Initialization of station system and identify station types.
        :param name:
        :param ssc:
        """

        self.name = name
        # form DataFrame from ndarray
        self.station = DataFrame(ssc['station'])
        # sort the DataFrame self.station by bus no. and reset index
        self.station.sort_values(by=['bus_i', 'node_i'], inplace=True)
        self.station.reset_index(drop=True, inplace=True)

        # set depot's power parameters to zero
        self.station.loc[self.station['station_type'] == 'depot',
                                                'max_p_mw':'qload_mvar'] = 0

        # find the index of depot and microgrid, ndarray
        self.idx_station = self.station.index.values
        self.idx_depot = self.station[self.station['station_type']
                                      == 'depot'].index.values
        self.idx_microgrid = self.station[self.station['station_type']
                                          == 'microgrid'].index.values

        # station's location in transportation network
        self.station_node = self.station.loc[:, 'node_i'].values
        # stations location in distribution system
        self.station_bus = self.station['bus_i'].values

        # no. of various types of stations
        self.n_station = self.idx_station.size
        self.n_depot = self.idx_depot.size
        self.n_microgrid = self.idx_microgrid.size

    def init_localload(self, time_sys, ds_sys):

        from pytess.load_info import init_load_profile

        self.p_localload, self.q_localload, _ = init_load_profile(
            load=self.station, time_sys=time_sys, ds_sys=ds_sys,
            P_LOAD='pload_mw', Q_LOAD='qload_mvar')

    def map_station2dsts(self, ds_sys, ts_sys):
        '''
        construct mapping of station to distribution bus and transportation node
        :return:
        '''

        from scipy.sparse import csr_matrix

        # # list(chain(*a)) convert list of list to list
        # # only update microgrids' bus, not including depots' bus
        # # it updates the microgrids' bus no. in current distribution network
        # self.station.loc[self.idx_microgrid, 'bus_i'] = [
        #     ds_sys.ppnet.bus.loc[
        #         ds_sys.ppnet.bus['e_name']==e_station, 'name'].tolist()[0]
        #     for e_station in self.station['bus_i'] if e_station >= 0]
        #
        # # Update station's node no. in current transportation network
        # self.station['node_i'] = [ts_sys.node.loc[
        #             ts_sys.node['e_node_i']==e_station, 'node_i'].tolist()[0]
        #     for e_station in self.station['node_i']]

        # extract idx_depot and idx_microgrid
        idx_depot = self.idx_depot
        idx_microgrid = self.idx_microgrid
        n_depot = self.n_depot
        n_microgrid = self.n_microgrid
        n_station = self.n_station

        # no. of station, bus and node
        n_station = self.n_station
        n_bus = ds_sys.n_bus
        n_node = ts_sys.n_node

        # each microgrid corresponds to each distribution bus
        # (n_station, n_bus)
        self.station2dsbus = csr_matrix(
            (ones(n_microgrid), (idx_microgrid, self.station.loc[idx_microgrid, 'bus_i'])),
            shape=(n_station, n_bus), dtype=int)

        # each station corresponds to each transportation node
        # (n_station, n_node)
        self.station2tsnode = csr_matrix(
            (ones(n_station), (range(n_station), self.station['node_i'])),
            shape=(n_station, n_node), dtype=int)

    def find_static_travel_time(self, ts_sys, tess_sys):
        '''
        Find the shortest path among all the microgrid and depot stations.
        :param ssnet:
        :return:
        '''
        from itertools import combinations, permutations
        import networkx as nx

        # ndarray to represent shortest path length, (n_station, n_station),
        # square matrix
        shortest_path_length_static = zeros((self.n_station, self.n_station))
        # only compute the upper triangle by combinations
        for e_s_node, e_t_node in combinations(self.idx_station, 2):
            shortest_path_length_static[e_s_node, e_t_node] = nx.shortest_path_length(
                G=ts_sys.graph, source=self.station.loc[e_s_node, 'node_i'],
                target=self.station.loc[e_t_node, 'node_i'], weight='length')
        # form the symmetric matrix from upper triangle
        shortest_path_length_static += shortest_path_length_static.T - np.diag(
            shortest_path_length_static.diagonal())

        # To set shortest path
        shortest_path_static = zeros((self.n_station, self.n_station),
                              dtype=object)
        # compute all the permutations
        for e_s_node, e_t_node in permutations(self.idx_station, 2):
            shortest_path_static[e_s_node, e_t_node] = nx.shortest_path(G=ts_sys.graph,
                source=self.station.loc[e_s_node, 'node_i'],
                target=self.station.loc[e_t_node, 'node_i'], weight='length')

        # Define travel_time_static for each tess, (n_tess, n_station, n_station)
        travel_time_static = np.zeros((tess_sys.n_tess, self.n_station, self.n_station),
                               dtype=int)

        # todo need to improve to multi-layer
        # The dots (...) represent as many colons as needed to produce a
        # complete indexing tuple.
        for i_tess in range(tess_sys.n_tess):
            travel_time_static[i_tess, ...] = np.ceil(shortest_path_length_static
                / tess_sys.tess['avg_v_km/h'][i_tess]).astype(int)

        # store in class
        self.shortest_path_length_static = shortest_path_length_static
        # store shortest_path_static
        self.shortest_path_static = shortest_path_static
        self.travel_time_static = travel_time_static
        # return path_length_table # it's view not copy

    def set_optimization_case(self, time_sys, ds_sys):

        n_interval = time_sys.n_interval
        # upper bounds of active/reactive power output of station
        # (including depot) of the shape of (n_station, 1)
        self.station_p_u = self.station['max_p_mw'][:, np.newaxis] \
                           / ds_sys.ppnet.sn_mva
        self.station_q_u = self.station['max_q_mvar'][:, np.newaxis] \
                           / ds_sys.ppnet.sn_mva

        # lower bounds of active/reactive power output of station
        # (n_station, 1)
        self.station_p_l = self.station['min_p_mw'][:, np.newaxis] \
                           / ds_sys.ppnet.sn_mva
        self.station_q_l = self.station['min_q_mvar'][:, np.newaxis] \
                           / ds_sys.ppnet.sn_mva

        # station energy capacity of station
        # The modification is moved to main funciton and here only retrieve the
        # data
        # (n_station, 1)

        self.station_e_u = self.station['cap_e_mwh'][:, np.newaxis] \
                           / ds_sys.ppnet.sn_mva

        self.station_e_l = self.station['min_r_mwh'][:, np.newaxis] \
                           / ds_sys.ppnet.sn_mva

        # station generation cost, (n_station, 1)
        self.station_gencost = self.station['cost_$/mwh'][:, np.newaxis]