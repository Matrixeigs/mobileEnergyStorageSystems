import numpy as np
from numpy import array, zeros, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import cplex as cpx


class TimeSpaceNetwork:
    '''

    '''

    def __init__(self, name):
        self.name = name

    # todo def __repr__(self):
    # def __repr__(self):  # pragma: no cover
    #     r = "This pandapower network includes the following parameter tables:"
    #     par = []
    #     res = []
    #     for tb in list(self.keys()):
    #         if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
    #             if 'res_' in tb:
    #                 res.append(tb)
    #             else:
    #                 par.append(tb)
    #     for tb in par:
    #         length = len(self[tb])
    #         r += "\n   - %s (%s %s)" % (
    #         tb, length, "elements" if length > 1 else "element")
    #     if res:
    #         r += "\n and the following results tables:"
    #         for tb in res:
    #             length = len(self[tb])
    #             r += "\n   - %s (%s %s)" % (
    #             tb, length, "elements" if length > 1 else "element")
    #     return r

    def set_tsn_model_new(self, time_sys, ds_sys, ts_sys, ss_sys, tess_sys):
        # this_interval = interval[0]


        # time info
        n_interval = time_sys.n_interval
        n_interval_window = time_sys.n_interval_window
        interval = time_sys.interval
        n_timepoint_window = time_sys.n_timepoint_window

        timepoint = time_sys.timepoint
        interval_window = time_sys.interval_window
        timepoint_window = time_sys.timepoint_window

        # station params
        idx_station = ss_sys.idx_station
        n_station = ss_sys.n_station
        n_microgrid = ss_sys.n_microgrid
        idx_microgrid = ss_sys.idx_microgrid
        n_depot = ss_sys.n_depot
        idx_depot = ss_sys.idx_depot
        station_node = ss_sys.station_node

        # tess info
        n_tess = tess_sys.n_tess
        travel_time = tess_sys.travel_time

        # define output lists
        self.tsn_node = []
        self.n_tsn_node = []

        self.tsn_arc = []
        self.n_tsn_arc = []

        self.tsn_cut_set = []
        self.tsn_holding_arc = []


        self.tsn_f2arc = []
        self.tsn_t2arc = []

        for i_tess in range(n_tess):

            travel_time_tess = travel_time[i_tess]

            # query tess current location
            current_location_tess = tess_sys.current_location[i_tess]
            # current_location_tess = 23

            # ------------------------------------------------------------------
            # set up tsn nodes
            # a table with 3 columns | # | t | location|

            # col_node_number = np.arange(n_microgrid*(n_timepoint-2)) + 1
            # col_node_number = col_node_number[:, np.newaxis]
            # col_node_time_point = np.array([i // n_microgrid for i in range(
            #     n_microgrid*(n_timepoint-2))]) + 1
            # col_node_time_point = col_node_time_point[:, np.newaxis]
            # col_node_location = np.array([idx_microgrid[i%n_microgrid] for i in
            #                               range(n_microgrid*(n_timepoint-2))])
            # col_node_location = col_node_location[:, np.newaxis]
            # # tsn node for each tess, will be override
            # tsn_node_tess = np.concatenate((col_node_number, col_node_time_point,
            #                            col_node_location), axis=1)

            # generate a tsn node table with 3 columns | # | t | location| for
            # time point window 1 to end-1, only including microgrids
            # column name constant
            self.NODE_I = 0
            self.NODE_T = 1
            self.NODE_L = 2

            # List comprehension to simplify the table counstruction
            tsn_node_tess = np.array([
                [i+1, i//n_microgrid+1, idx_microgrid[i%n_microgrid]]
                for i in range(n_microgrid*(n_timepoint_window-2))
                               ], dtype=int)

            # if the tsn_node_tess is empty, indicating the last interval in the
            # entire horizon
            if not tsn_node_tess.size:
                tsn_node_tess = np.zeros((0, 3), dtype=int)

            # add tsn node for the time point 0, based on current location
            if current_location_tess in station_node:
                # the tess is at station
                # assert the travel_time square array should match square of n_station
                assert travel_time_tess.size == n_station ** 2
                # which station is the current location
                temp_index = np.flatnonzero(np.array(station_node) == current_location_tess)

            else:
                # assert the travel_time square array should match square of n_station
                assert travel_time_tess.size == (n_station + 1) ** 2
                # the extra node is always put at the last of travel_time array.
                temp_index = -1
            # There is always only one row of the initial tsn node indicating
            # current location

            adding_nodes = np.array([0, 0, temp_index]).reshape(1, -1)
            # try:
            tsn_node_tess = np.concatenate((adding_nodes, tsn_node_tess), axis=0)
            # except:
            #     pass

            # adding the ending node, the depot nodes at the last time point if
            # the time point is the last time point of the entire time horizon.
            if interval[-1] == n_interval - 1:
                adding_nodes = np.array([[tsn_node_tess[-1, 0] + i + 1, n_timepoint_window - 1,
                    idx_depot[i]] for i in range(n_depot)])
            else:
                adding_nodes = np.array([[tsn_node_tess[-1, 0] + i + 1, n_timepoint_window - 1,
                    idx_microgrid[i]] for i in range(n_microgrid)])

            tsn_node_tess = np.concatenate((tsn_node_tess, adding_nodes))

            # number of tsn node for this tess
            n_tsn_node_tess = tsn_node_tess.shape[0]
            # append tsn node table
            self.n_tsn_node.append(n_tsn_node_tess)
            self.tsn_node.append(tsn_node_tess)

            # ------------------------------------------------------------------
            # for tsn arcs
            self.ARC_I = 0
            self.S_NODE_I = 1
            self.T_NODE_I = 2
            self.S_NODE_T = 3
            self.S_NODE_L = 4
            self.T_NODE_T = 5
            self.T_NODE_L = 6
            self.ARC_LEN = 7
            # set up tsn arcs for each tess of 8 columns, will be override
            tsn_arc_tess = np.zeros((0, 8), dtype=int)
            i_arc = 0
            for i_node_number, i_node_t, i_node_location in tsn_node_tess:
                for j_node_location in idx_station:
                    if i_node_location != j_node_location:
                        try:
                            arc_length = travel_time[i_tess][i_node_location, j_node_location]
                        except:
                            pass
                    else:
                        # if they are the same location, holding arc
                        arc_length = 1

                    j_node_t = i_node_t + arc_length
                    # find the j_node_number through j_node_t and j_node_location
                    # The j_node_number is either a one element array or empty array
                    j_node_number = tsn_node_tess[(tsn_node_tess[:, 1] == j_node_t)
                        & (tsn_node_tess[:, 2] == j_node_location), 0]

                    if j_node_number.size:
                        # if j_node_number is not empty, indicating we can find
                        # the target node in the node table, so add the arc to arc table
                        new_arc = np.array([i_arc, i_node_number,
                            j_node_number.item(), i_node_t,
                            i_node_location, j_node_t, j_node_location,
                            arc_length]).reshape(1, -1)

                        i_arc += 1

                        tsn_arc_tess = np.concatenate((tsn_arc_tess, new_arc), axis=0)

            # number of tsn arcs for this tess
            n_tsn_arc_tess = tsn_arc_tess.shape[0]
            # check the number
            try:
                assert n_tsn_arc_tess == tsn_arc_tess[-1, self.ARC_I] + 1
            except:
                pass

            # append the tsn arc table
            self.n_tsn_arc.append(n_tsn_arc_tess)
            self.tsn_arc.append(tsn_arc_tess)

            # ------------------------------------------------------------------
            # cut set for each interval for each tess, (n_tsn_arc_tess, n_interval_window)
            tsn_cut_set_tess = np.zeros((n_tsn_arc_tess, n_interval_window),
                                        dtype=bool)

            for j_interval_window in interval_window:
                # & bit-operators, parathesis is compulsory
                # a cut-set for each interval, involving all the arcs crossing
                # the cut in this j_interval_window
                tsn_cut_set_tess[:, j_interval_window] = (
                    tsn_arc_tess[:, self.S_NODE_T] <= j_interval_window) & \
                    (tsn_arc_tess[:, self.T_NODE_T] > j_interval_window)

            self.tsn_cut_set.append(tsn_cut_set_tess)

            # ------------------------------------------------------------------
            # holding arc for each arc for each tess, (n_tsn_arc_tess, n_interval_window)
            tsn_holding_arc_tess = np.zeros((n_tsn_arc_tess, n_interval_window),
                                            dtype=bool)

            for j_interval_window in interval_window:
                tsn_holding_arc_tess[:, j_interval_window] = \
                    (tsn_arc_tess[:, self.S_NODE_T] == j_interval_window) & \
                    (tsn_arc_tess[:, self.T_NODE_T] == j_interval_window + 1) & \
                    (tsn_arc_tess[:, self.S_NODE_L] == tsn_arc_tess[:, self.T_NODE_L])

            self.tsn_holding_arc.append(tsn_holding_arc_tess)

            # ------------------------------------------------------------------
            # map arc's from node for each tess, (n_tsn_node_tess,  n_tsn_arc_tess)
            tsn_f2arc_tess = csr_matrix((np.ones(n_tsn_arc_tess),
                            (tsn_arc_tess[:, self.S_NODE_I], range(n_tsn_arc_tess))),
                           shape=(n_tsn_node_tess, n_tsn_arc_tess), dtype=int)

            self.tsn_f2arc.append(tsn_f2arc_tess)

            # ------------------------------------------------------------------
            # map arc's to node for each tess, (n_tsn_node_tess,  n_tsn_arc_tess)
            tsn_t2arc_tess = csr_matrix((np.ones(n_tsn_arc_tess),
                            (tsn_arc_tess[:, self.T_NODE_I], range(n_tsn_arc_tess))),
                           shape=(n_tsn_node_tess, n_tsn_arc_tess), dtype=int)

            self.tsn_t2arc.append(tsn_t2arc_tess)

            pass

    def set_tsn_model(self, ds_sys, ts_sys, ss_sys, tess_sys):
        """Formation of time-space network, referring to model and figure in
        paper
        :param tsc:
        :param n_interval:
        :return:
        """
        # temporary parameters for tsc
        from itertools import permutations
        from scipy.sparse import csr_matrix, lil_matrix

        travel_time = ss_sys.travel_time
        # extract time interval
        n_interval = ds_sys.n_interval
        n_timepoint = n_interval + 1
        # no. of station on map
        n_station = ss_sys.n_station
        # identify depot and microgrid
        idx_depot = ss_sys.idx_depot
        idx_microgrid = ss_sys.idx_microgrid
        n_depot = ss_sys.n_depot
        n_microgrid = ss_sys.n_microgrid

        # The number of tsn nodes in tsn graph
        n_tsn_node_source = n_depot  # For source node at the time point 0
        n_tsn_node_microgrid = n_microgrid * (n_timepoint-2)
        n_tsn_node_sink = n_depot  # For source node at the the last time point
        # total tsn_node
        n_tsn_node = n_tsn_node_source + n_tsn_node_microgrid + n_tsn_node_sink

        # indexing of tsn nodes, ndarray(n_station, n_timepoint)
        # in this array, (i, j) indicates the tsn node for i-th station at
        # time point j, if value >= 0, if value < 0, meaning there is no
        # associated tsn node, so the - 10 is for this reason.
        tsn_node = zeros((n_station, n_timepoint), dtype=int) - 10
        # at time point 0
        tsn_node[idx_depot, 0] = arange(n_tsn_node_source)
        # from time point 1 to second-to-last
        tsn_node[idx_microgrid, 1:(n_timepoint-1)] = arange(n_tsn_node_source,
                              n_tsn_node_source+n_tsn_node_microgrid).reshape(
            n_microgrid, n_timepoint-2, order='F')
        # at the last time point
        tsn_node[idx_depot, -1] = arange(n_tsn_node-n_tsn_node_sink,n_tsn_node)

        # To form multi-layer to (n_tess, n_station, n_timepoint)
        tsn_node = np.tile(tsn_node, reps=(tess_sys.n_tess, 1, 1))

        # Set up ndarray (??, 3) indicating tsn arcs
        #  [from_tsn_node to_tsn_node travel time]
        # column index
        F_TSN_NODE = 0
        T_TSN_NODE = 1
        TRAVEL_TIME = 2

        n_tess = tess_sys.n_tess

        # for source arcs, reshape (n_tess * n_depot * n_microgrid, 3) into
        # (n_tess, n_depot * n_microgrid, 3)
        tsn_arc_source = np.array([
            [tsn_node[i_tess, e_depot, 0],
             tsn_node[i_tess, e_microgrid,
                      0 + travel_time[i_tess, e_depot, e_microgrid]],
             travel_time[i_tess, e_depot, e_microgrid]]
            for i_tess in range(n_tess)
            for e_depot in idx_depot
            for e_microgrid in idx_microgrid]).reshape(n_tess, -1, 3)
            # -1 means inferred from other axis

        # for sink arcs
        tsn_arc_sink = np.array([
            [tsn_node[i_tess, e_microgrid,
                      -1 - travel_time[i_tess, e_microgrid, e_depot]],
             tsn_node[i_tess, e_depot, -1],
             travel_time[i_tess, e_microgrid, e_depot]]
            for i_tess in range(n_tess)
            for e_depot in idx_depot
            for e_microgrid in idx_microgrid]).reshape(n_tess, -1, 3)

        # fro normal arcs (transportation among microgrids)
        # tsn_arc_normal = np.zeros((0, 3))

        # for j_timepoint in range(1, n_timepoint-2):
        # tsn_arc lists tsn arcs for each layer,
        # (n_tess) list of (n_unkonw, 3) array
        tsn_arc = []
        n_tsn_arc = np.zeros(n_tess, dtype=int)

        for i_tess in range(n_tess):

            # all the moving arcs for i_tess-th layer tsn, (n_unkown, 3)
            tsn_arc_moving_temp = np.array([
                [tsn_node[i_tess, s_microgrid, j_timepoint],
                 tsn_node[i_tess, t_microgrid,
                          j_timepoint+travel_time[i_tess, s_microgrid, t_microgrid]],
                 travel_time[i_tess, s_microgrid, t_microgrid]]
                for j_timepoint in range(1, n_timepoint-2)
                for s_microgrid, t_microgrid in permutations(idx_microgrid, 2)
                if j_timepoint+travel_time[i_tess, s_microgrid, t_microgrid]
                   <= n_timepoint-2])

            # all the holding arcs for i_tess-th layer tsn, (n_unkown, 3)
            tsn_arc_holding_temp = np.array([
                [tsn_node[i_tess, e_microgrid, j_timepoint],
                 tsn_node[i_tess, e_microgrid, j_timepoint+1],
                 0]
                for j_timepoint in range(1, n_timepoint-2)
                for e_microgrid in idx_microgrid])

            # todo is it possible to have void set for tsn_arc_moving_temp?
            tsn_arc.append(np.vstack([tsn_arc_source[i_tess, :, :],
                                            tsn_arc_moving_temp,
                                            tsn_arc_holding_temp,
                                             tsn_arc_sink[i_tess, :, :]]))

            n_tsn_arc[i_tess] = tsn_arc[i_tess].shape[0]

            # if tsn_arc_moving_temp.shape[0]:
            #     tsn_arc_normal = np.vstack([tsn_arc_normal,
            #                         tsn_arc_moving_temp, tsn_arc_holding_temp])
            # else:
            #     tsn_arc_normal = np.vstack([tsn_arc_normal,
            #                                 tsn_arc_holding_temp])
        # Consolidate all arcs
        # tsn_arc = np.vstack([tsn_arc_source, tsn_arc_normal, tsn_arc_sink])

        # n_tsn_arc = tsn_arc.shape[0]
        # index of the parking arc position in tsn_arc_table,
        # (n_station, n_interval)

        # holding arc for each layer
        tsn_arc_holding = []
        # cut set for each layer
        tsn_cut_set = []
        # tsn_arc_holding = zeros((n_tess, n_tsn_arc, n_interval), dtype=bool)

        for i_tess in range(n_tess):
            idx_tsn_arc_holding_temp = zeros((n_station, n_interval), dtype=int)
            idx_tsn_arc_holding_temp[idx_microgrid, 1:-1] = np.flatnonzero(
                tsn_arc[i_tess][:, TRAVEL_TIME] == 0).reshape(
                (n_microgrid, n_interval-2), order='F')

            # convert to True/False matrix to indicate holding arcs
            # at each interval, (n_tsn_arc[i_tess], n_interval)
            #todo transform to sparse matrix
            tsn_arc_holding_temp = zeros((n_tsn_arc[i_tess],
                                          n_interval), dtype=bool)

            for j_interval in range(1, n_interval-1):
                tsn_arc_holding_temp[
                    idx_tsn_arc_holding_temp[idx_microgrid, j_interval],
                    j_interval] = 1

            tsn_arc_holding.append(tsn_arc_holding_temp)

            # ------------------------------------------------------------
            # tsn_cut_set matrix
            tsn_cut_set_temp = zeros((n_tsn_arc[i_tess], n_interval),
                                     dtype=bool)

            for j_interval in range(n_interval):
                # & bit-operators, parathesis is compulsory
                # a cut-set for each interval
                tsn_cut_set_temp[:, j_interval] = (tsn_arc[i_tess]
                    [:, F_TSN_NODE] <= max(tsn_node[i_tess, :, j_interval])) & (
                        tsn_arc[i_tess][:, T_TSN_NODE] > max(tsn_node[
                                                    i_tess, :, j_interval]))

            tsn_cut_set.append(tsn_cut_set_temp)
            # tsn_cut_set = lil_matrix(tsn_cut_set)  #lil_matrix causes problem
            # with value assignment to ndarray

            # mapping_tsn_f,t matrix (n_tsn_node, n_tsn_arc), indicate each
            # arc's from bus and to bus
        mapping_tsn_f2arc = [csr_matrix((ones(n_tsn_arc[i_tess]),
            (tsn_arc[i_tess][:, F_TSN_NODE], range(n_tsn_arc[i_tess]))),
            shape=(n_tsn_node, n_tsn_arc[i_tess]), dtype=int)
            for i_tess in range(n_tess)]

        mapping_tsn_t2arc = [csr_matrix((ones(n_tsn_arc[i_tess]),
            (tsn_arc[i_tess][:, T_TSN_NODE], range(n_tsn_arc[i_tess]))),
            shape=(n_tsn_node, n_tsn_arc[i_tess]), dtype=int)
            for i_tess in range(n_tess)]

        # mapping_tsn_t2arc = csr_matrix(
        #     (ones(n_tsn_arc), (tsn_arc[:, T_TSN_NODE], range(n_tsn_arc))),
        #     shape=(n_tsn_node, n_tsn_arc), dtype=int)

        # store in class
        self.n_tsn_node_source = n_tsn_node_source
        self.n_tsn_ndoe_microgrid = n_tsn_node_microgrid
        self.n_tsn_node_sink = n_tsn_node_sink
        self.n_tsn_node = n_tsn_node
        self.tsn_node = tsn_node
        self.tsn_arc_source = tsn_arc_source
        self.tsn_arc_sink = tsn_arc_sink
        self.tsn_arc = tsn_arc
        # self.idx_tsn_arc_holding = idx_tsn_arc_holding
        self.tsn_arc_holding = tsn_arc_holding
        self.n_tsn_arc = n_tsn_arc
        self.tsn_cut_set = tsn_cut_set
        self.mapping_tsn_f2arc = mapping_tsn_f2arc
        self.mapping_tsn_t2arc = mapping_tsn_t2arc

    def set_stationary_solution(self, ds_sys, ts_sys, ss_sys, tess_sys):
        '''

        :param ds_sys:
        :param ts_sys:
        :param ss_sys:
        :param tess_sys:
        :return:
        '''

        # retrieve parameters
        tsn_node = self.tsn_node
        tsn_arc = self.tsn_arc
        n_tsn_arc = self.n_tsn_arc
        tsn_arc_source = self.tsn_arc_source
        tsn_arc_sink = self.tsn_arc_sink
        idx_microgrid = ss_sys.idx_microgrid
        idx_depot = ss_sys.idx_depot
        n_tess = tess_sys.n_tess
        n_microgrid = ss_sys.n_microgrid

        # To save the solution for tess staying at the specific microgrid
        self.stationary_solution = []

        for i_tess in range(n_tess):
            stationary_solution_temp = np.zeros((n_microgrid,
                                            n_tsn_arc[i_tess]), dtype=bool)

            for i_microgrid in range(n_microgrid):
                # The column 1 indicates the source arc to the microgrid node
                first_microgrid_node = tsn_arc_source[i_tess, i_microgrid, 1]
                # The column 0 indicates the sink arc from the microgird node
                last_microgrid_node = tsn_arc_sink[i_tess, i_microgrid, 0]

                # Identify the associated node
                # microgrid nodes
                active_node = tsn_node[i_tess,
                    idx_microgrid[i_microgrid], np.logical_and(
                    tsn_node[i_tess, idx_microgrid[i_microgrid],
                    :] >= first_microgrid_node,
                    tsn_node[i_tess, idx_microgrid[i_microgrid],
                    :] <= last_microgrid_node)]
                # Append depot nodes
                # todo if there are more depots, indicate which one is initial
                active_node = np.hstack([tsn_node[i_tess, idx_depot[0], 0],
                            active_node, tsn_node[i_tess, idx_depot[0], -1]])

                # Identify the associated arc
                active_arc = np.hstack([active_node[:-1, np.newaxis],
                                        active_node[1:, np.newaxis]])

                # To find the index of active arcs in all tsn arcs
                tsn_arc_list = tsn_arc[i_tess][:, :-1].tolist()
                active_arc_list = active_arc.tolist()
                # the index
                idx_active_arc = [tsn_arc_list.index(element)
                                  for element in active_arc_list]

                # convet index to boolean array
                stationary_solution_temp[i_microgrid, idx_active_arc] = 1

            self.stationary_solution.append(stationary_solution_temp)

        pass


if __name__ == '__main__':

    tsn_sys = TimeSpaceNetwork('TSN')
    tsn_sys.set_tsn_model_new(ds_sys=1, ts_sys=1, ss_sys=1, tess_sys=1)