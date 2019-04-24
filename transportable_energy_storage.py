import numpy as np
from numpy import array, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
import networkx as nx

import cplex as cpx

class TransportableEnergyStorage:

    def __init__(self, name, tessc):
        # instance's name
        self.name = name
        # form DataFrame from ndarray
        self.tess = DataFrame(tessc['tess'])
        # The no. of tess
        self.n_tess = self.tess.shape[0]

    def init_graph_from_ts(self, ts_sys):
        # avoid this as each element has the same id
        # self.tess_graph = [nx.Graph(ts_sys.graph)] * self.n_tess
        # each element is different
        self.tess_graph = [nx.Graph(ts_sys.graph) for i in range(self.n_tess)]

        return self.tess_graph

    def find_this_current_location(self, time_sys, ds_sys, ss_sys, tess_sys,
                                   tsn_sys, this_graph, i_tess, action_arc=[]):
        """
        :param ds_sys
        :param action_arc: for each tess's arcs in the tsn node based on this_graph]
        :return: this_current_location:, tuple (u, v, distance_to_u, distance_to_v)
        or existing node number, number is in the transportation network
        """
        # this_current_location = []
        delta_t = time_sys.delta_t
        station_node = ss_sys.station_node
        interval = time_sys.interval

        if interval[0]:
            # if it's not the initial interval in the entire horizon, it needs to
            # get the previous arc to find the current location
            assert len(action_arc), "It's not the initial interval in the entire horizon, " \
                                 "action_arc information cannot be empty"
            # query status from arc
            # the orgin node number in station system
            origin = action_arc[tsn_sys.S_NODE_L]
            # the destination node number in transportation network
            destination = action_arc[tsn_sys.T_NODE_L]
            arc_length = action_arc[tsn_sys.ARC_LEN]

            if arc_length > 1:
                # Find the shortest path and path
                path = self.shortest_path[i_tess][origin, destination]  #  [0, 1, 25, 23] a series of nodes
                # path_length = self.shortest_path_length[i_tess][origin, destination]
                # [10, 15, 30] length of each path segment
                path_length = [self.tess_graph[i_tess].edges[path[i],
                    path[i+1]]['length']  for i in range(len(path)-1)]

                travel_distance = tess_sys.tess['avg_v_km/h'][i_tess] * delta_t
                # segment starts with 0
                segment = np.concatenate((np.zeros(1), np.cumsum(path_length)))  # [0, 10, 25, 55]

                if not travel_distance in segment:
                    # on the edge, need to add new node
                    landing_segment = np.searchsorted(segment, travel_distance)

                    # If not exceeding the path range
                    # landing_segment = np.argsort(np.concatenate(
                    #     (segment, np.array([travel_distance]))))[-1]
                    # node number in the transportation network
                    u = path[landing_segment - 1]
                    v = path[landing_segment]
                    #
                    distance2u = travel_distance - segment[landing_segment - 1]
                    distance2v = segment[landing_segment] - travel_distance

                    if u == 'moving node':
                        # if u is still 'moving node', it needs to extend to another node
                        # through the edge connecting moving node

                        # find the two nodes connected to moving node, one is v, another
                        # is goal. find neighbors
                        # u is a int converted from a single-element list
                        u = [n for n in this_graph['moving node'] if n != v][0]
                        try:
                            distance2u += this_graph.edges['moving node', u]['length']
                        except:
                            pass

                    this_current_location = [u, v, distance2u, distance2v]

                else:
                    #  at existing node, no need to add new node
                    # the current location is the node number in the transporation
                    # network
                    idx_node = np.flatnonzero(segment == travel_distance).item()
                    this_current_location = path[idx_node]
            else:
                # if the arc length is only one, it only needs to get the
                # destination as current_location, the number needs to be
                # converted from station system into the transportation network
                this_current_location = ss_sys.station_node[destination]

        else:
            # if it's the initial interval in the entire horizon, get the initial
            # location
            this_current_location = tess_sys.tess.loc[i_tess, 'init_location']

        return this_current_location

    def find_travel_time(self, time_sys, ds_sys, ts_sys, ss_sys, tess_sys, tsn_sys, result_list):
        '''
        Find the shortest path among all the microgrid and depot stations.
        :param ssnet:
        :return:
        '''
        from itertools import combinations, permutations

        n_tess = self.n_tess

        delta_t = time_sys.delta_t
        interval = time_sys.interval

        # Create empty list
        shortest_path = []
        shortest_path_length = []
        travel_time = []
        current_location = []

        for i_tess in range(n_tess):
            # First find the current location based on previous graph
            # todo Need to incorporate tsn arc result_list to indicate the dispathcing
            #  and get the current location
            if interval[0]:
                # if it's not initialization
                # try:
                j_interval_previous = interval[0] - 1
                action_arc = result_list[j_interval_previous].realres_tess_arc_x[i_tess]
                current_location.append(self.find_this_current_location(time_sys=time_sys,
                    ds_sys=ds_sys, ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys,
                    this_graph=self.tess_graph[i_tess], i_tess=i_tess, action_arc=action_arc))
                # except:
                    # Simulate initial condition
                    # current_location.append(self.tess['init_location'].values[i_tess])
                    # Simulate adding new node
                    # current_location.append((3, 10, 16, 2))
                    # Simulate that tess stay at transportation node
                    # current_location.append(16)
            else:
                # it's the first interval in the entire horizon
                current_location.append(self.tess['init_location'][i_tess])

            # each tess owns a individual graph, it's deep copy
            self.tess_graph[i_tess] = nx.Graph(ts_sys.graph)
            this_tess_graph = self.tess_graph[i_tess]

            # ndarray to represent shortest path length, (n_station, n_station),
            # square matrix

            # station's node at transportation network
            combination_node = ss_sys.station_node.tolist()

            if isinstance(current_location[i_tess], list):
                # it needs to add new node in tranportation network
                u, v, distance2u, distance2v = current_location[i_tess]
                # add a temp node
                this_tess_graph.add_node('moving node')
                # add corresponding new edges
                this_tess_graph.add_weighted_edges_from(
                    [('moving node', u, distance2u), ('moving node', v, distance2v)],
                    weight='length')
                # remove the edges otherwise there is a loop
                this_tess_graph.remove_edge(u, v)

                combination_node.append('moving node')

                current_location[i_tess] = 'moving node'

            else:
                # if there is no added node in transportation network
                if current_location[i_tess] in combination_node:
                # if current location is at a station
                #     combination_node.append(current_location[i_tess])
                    pass
                else:
                    combination_node.append(current_location[i_tess])

            n_combination_node = len(combination_node)

            this_shortest_path_length = np.zeros((n_combination_node, n_combination_node))
            # only compute the upper triangle by combinations
            for i_s_node, i_t_node in combinations(range(n_combination_node), 2):
                this_shortest_path_length[i_s_node, i_t_node] = nx.shortest_path_length(
                    G=this_tess_graph, source=combination_node[i_s_node],
                    target=combination_node[i_t_node], weight='length')
            # form the symmetric matrix from upper triangle
            this_shortest_path_length += this_shortest_path_length.T - np.diag(
                this_shortest_path_length.diagonal())

            shortest_path_length.append(this_shortest_path_length)

            # To set shortest path
            this_shortest_path = np.zeros((n_combination_node, n_combination_node),
                                  dtype=object)
            # compute all the permutations
            # for i_s_node, i_t_node in permutations(range(n_combination_node), 2):
            # since permutation will ignore the diagonal element, leaving it zero,
            # this will incur error for latter use in optimization_result.py
            # Thus, use double for-loop to traverse all the pairs
            for i_s_node in range(n_combination_node):
                for j_t_node in range(n_combination_node):
                    this_shortest_path[i_s_node, j_t_node] = nx.shortest_path(G=this_tess_graph,
                        source=combination_node[i_s_node],
                        target=combination_node[j_t_node], weight='length')

            shortest_path.append(this_shortest_path)

            # Get travel_time_static for each tess, (n_combination_node, n_combination_node)
            this_travel_time= np.ceil(this_shortest_path_length
                    / self.tess['avg_v_km/h'][i_tess] / delta_t).astype(int)

            travel_time.append(this_travel_time)

        # store in class

        self.shortest_path_length = shortest_path_length

        # store shortest_path_static

        self.shortest_path = shortest_path

        self.travel_time = travel_time

        # i-th tess 'moving node' location is determined by self.tess_graph[i_tess]
        self.current_location = current_location

        pass

        # return path_length_table # it's view not copy

    def set_optimization_case(self, ds_sys, ss_sys):

        # upper and lower bounds for tess's charging/discharging power
        # (n_tess, 1)
        self.tess_pch_u = self.tess['ch_p_mw'][:,
                          np.newaxis] / ds_sys.sn_mva
        self.tess_pdch_u = self.tess['dch_p_mw'][:,
                           np.newaxis] / ds_sys.sn_mva

        # tess' energy capacity
        # (n_tess, 1)
        self.tess_cap_e = self.tess['cap_e_mwh'][:,
                          np.newaxis] / ds_sys.sn_mva

        # tess's initial energy
        # (n_tess, 1)
        self.tess_e_init = self.tess_cap_e * self.tess['init_soc'][:,
                                             np.newaxis]
        # tess's energy upper/lower bounds
        # (n_tess, 1)
        self.tess_e_u = self.tess_cap_e * self.tess['max_soc'][:, np.newaxis]
        self.tess_e_l = self.tess_cap_e * self.tess['min_soc'][:, np.newaxis]

        # charging/discharging efficiency
        self.tess_ch_efficiency = self.tess['ch_efficiency'][:, np.newaxis]
        self.tess_dch_efficiency = self.tess['dch_efficiency'][:, np.newaxis]

        # The battery maintenance  and transportation cost (n_tess, 1)
        self.tess_cost_power = self.tess['cost_power'][:, np.newaxis]

        self.tess_cost_transportation = self.tess['cost_transportation'][:,
                                        np.newaxis]

        # Indicate tess's initial position in transportation system, (n_tess, )
        idx_tess_init_location = array([np.flatnonzero(
            ss_sys.station['node_i'] == self.tess['init_location'][i_tess])
             for i_tess in range(self.n_tess)]).ravel()
        # mapping of each tess's initial location to transportation node
        # (n_tess, n_station)
        self.tess_init2tsnode = csr_matrix(
            (ones(self.n_tess), (range(self.n_tess), idx_tess_init_location)),
            shape=(self.n_tess, ss_sys.n_station))


if __name__ == "__main__":
    # pass

    tess_sys = TransportableEnergyStorage('nnn')

    tess_sys.find_this_current_location()
    pass