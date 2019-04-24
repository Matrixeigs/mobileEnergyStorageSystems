import numpy as np
from numpy import array, zeros, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import cplex as cpx


class OptimizationResult():

    def __init__(self, name):
        self.name = name

    def sort_this_result(self, model, time_sys, ds_sys, tess_sys, tsn_sys):
        '''
        :param self:
        :return:
        '''

        import copy
        # Store these for for each interval at the beginning of each interval
        self.tess_graph = copy.deepcopy(tess_sys.tess_graph)
        # i-th tess 'moving node' location is determined by self.tess_graph[i_tess]
        self.current_location = copy.deepcopy(tess_sys.current_location)
        self.tsn_node = copy.deepcopy(tsn_sys.tsn_node)
        self.tsn_arc = copy.deepcopy(tsn_sys.tsn_arc)
        self.shortest_path = copy.deepcopy(tess_sys.shortest_path)
        self.shortest_path_length = copy.deepcopy(tess_sys.shortest_path_length)
        self.travel_time = copy.deepcopy(tess_sys.travel_time)

        self.res_status = model.solution.status[model.solution.get_status()]
        self.res_goal = model.solution.get_objective_value()

        # coloumn vector for solutions
        values = array(model.solution.get_values())

        # --------------------------------------------------------------------------
        # tess results
        # charging power from station to tess at interval t,
        # (n_tess, n_station, n_interval_window)
        self.res_tess2st_pch_x = values[model.var_tess2st_pch_x]
        # (n_tess, n_interval_window)
        self.res_tess_pch_x = self.res_tess2st_pch_x.sum(axis=1)
        # (n_station, n_interval_window, indicating each station's sending power at
        # interval t
        self.res_station_outp_x = self.res_tess2st_pch_x.sum(axis=0)

        # discharging power from tess to station at time span t,
        # (n_tess, n_station, n_interval_window)
        self.res_tess2st_pdch_x = values[model.var_tess2st_pdch_x]
        # (n_tess, n_interval_window)
        self.res_tess_pdch_x = self.res_tess2st_pdch_x.sum(axis=1)
        # (n_station, n_interval_window), indicating each station's receiving power
        # at interval t.
        self.res_station_inp_x = self.res_tess2st_pdch_x.sum(axis=0)

        # (n_tess, n_interval_window), tess's net output power to mg
        self.res_tess_netoutp_x = self.res_tess_pdch_x - self.res_tess_pch_x
        # (n_station, n_interval_window),
        # indicating each station's net receiving power
        # associated with tess at interval t.
        self.res_station_netinp_x = self.res_station_inp_x - self.res_station_outp_x

        # tess's energy at interval t, (n_tess, n_interval_window)
        self.res_tess_e_x = values[model.var_tess_e_x]
        # charging sign for tess at interval t, (n_tess, n_interval_window)
        self.res_sign_ch_x = values[model.var_sign_ch_x]
        # discharging sign for tess at interval t, (n_tess, n_interval_window)
        self.res_sign_dch_x = values[model.var_sign_dch_x]
        # arc status for tess,  (n_tess) list of boolean (n_tsn_arc[i_tess], )
        n_tess = tess_sys.n_tess
        # Important! astype(bool)
        self.res_tess_arc_x = [values[model.var_tess_arc_x[i_tess]].astype(bool)
                               for i_tess in range(n_tess)]
        # only take the first arc to implement
        self.res_action_arcs = [tsn_sys.tsn_arc[i_tess][self.res_tess_arc_x[i_tess]][0]
                            for i_tess in range(n_tess)]
        # self.res_tess_arc_x = values[model.var_tess_arc_x]
        # generate tsn routes from tsn arcs
        self.gen_tsn_route(tess_sys=tess_sys, tsn_sys=tsn_sys)

        # Transportation status for tess at interval t,  (n_tess, n_interval_window)
        self.res_sign_onroad_x = values[model.var_sign_onroad_x]

        # --------------------------------------------------------------------------
        # Station results
        # active power output of station, (n_station, n_interval_window)
        self.res_station_p_x = values[model.var_station_p_x]
        # reactive power output of station, (n_station, n_interval_window)
        self.res_station_q_x = values[model.var_station_q_x]
        # the amount of energy of station, (n_station, n_interval_window)
        self.res_station_e_x = values[model.var_station_e_x]

        # --------------------------------------------------------------------------
        # distribution system results
        # line active power, (n_line, n_interval_window)
        self.res_pij_x = values[model.var_pij_x]
        # line reactive power, (n_line, n_interval_window)
        self.res_qij_x = values[model.var_qij_x]
        # bus voltage, (n_bus, n_interval_window)
        self.res_vm_x = values[model.var_vm_x]
        # aggregated active power generation at DS level,
        # (n_station, n_interval_window)
        self.res_aggregate_pg_x = values[model.var_aggregate_pg_x]
        # aggregated reactive power generation at DS level,
        # (n_station, n_interval_window)
        self.res_aggregate_qg_x = values[model.var_aggregate_qg_x]
        # # sign for load restoration, (n_load, n_interval_window)
        # self.res_load_x = values[self.var_gama_load_x]
        # load restoration
        self.res_pd_x = values[model.var_pd_x]  # (n_load, n_interval_window)
        self.res_qd_x = values[model.var_qd_x]  # (n_load, n_interval_window)
        # (n_load, n_interval_window)
        interval = time_sys.interval
        self.res_pdcut_x = ds_sys.pload[:, interval] - self.res_pd_x
        # (n_load, n_interval_window)
        self.res_qdcut_x = ds_sys.qload[:, interval] - self.res_qd_x
        # Line connection status, (n_line)
        self.res_alpha_branch_x = values[model.var_alpha_branch_x]
        # Auxiliary variables for line status,  (n_line)
        self.res_betaij_x = values[model.var_betaij_x]
        self.res_betaji_x = values[model.var_betaji_x]

    def sort_this_result_2stage(self, model, time_sys, scenario_sys, ds_sys, ss_sys, tess_sys, tsn_sys):
        '''
        :param self:
        :return:
        '''

        import copy
        # Store these for for each interval at the beginning of each interval
        self.tess_graph = copy.deepcopy(tess_sys.tess_graph)
        # i-th tess 'moving node' location is determined by self.tess_graph[i_tess]
        self.current_location = copy.deepcopy(tess_sys.current_location)
        self.tsn_node = copy.deepcopy(tsn_sys.tsn_node)
        self.tsn_arc = copy.deepcopy(tsn_sys.tsn_arc)
        self.shortest_path = copy.deepcopy(tess_sys.shortest_path)
        self.shortest_path_length = copy.deepcopy(tess_sys.shortest_path_length)
        self.travel_time = copy.deepcopy(tess_sys.travel_time)


        self.res_status = model.solution.status[model.solution.get_status()]
        self.res_goal = model.solution.get_objective_value()

        # coloumn vector for solutions
        values = array(model.solution.get_values())

        # decision in irst is the same, only take one to proceed
        THIS_SCENARIO = 0
        THIS_INTERVAL = 0

        # --------------------------------------------------------------------------
        # tess results
        # charging power from station to tess at interval t,
        # a list (n_scenario) of (n_tess, n_station, n_interval_window)
        n_scenario = scenario_sys.n_scenario
        n_station = ss_sys.n_station
        n_tess = tess_sys.n_tess
        self.res_tess2st_pch_x = [values[model.var_tess2st_pch_x[s_scenario]]
                                  for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario-1):
            assert (self.res_tess2st_pch_x[s_scenario][..., THIS_INTERVAL] ==
                   self.res_tess2st_pch_x[s_scenario+1][..., THIS_INTERVAL]).all()
        # (n_tess, n_station)
        self.realres_tess2st_pch_x = self.res_tess2st_pch_x[THIS_SCENARIO][..., THIS_INTERVAL]
        assert self.realres_tess2st_pch_x.shape == (n_tess, n_station)

        # (n_tess, n_interval_window)
        # self.res_tess_pch_x = self.res_tess2st_pch_x.sum(axis=1)
        # (n_station, n_interval_window, indicating each station's sending power at
        # interval t
        # self.res_station_outp_x = self.res_tess2st_pch_x.sum(axis=0)

        # discharging power from tess to station at time span t,
        # (n_tess, n_station, n_interval_window)
        self.res_tess2st_pdch_x = [values[model.var_tess2st_pdch_x[s_scenario]]
                                   for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario-1):
            assert (self.res_tess2st_pdch_x[s_scenario][..., THIS_INTERVAL] ==
                   self.res_tess2st_pdch_x[s_scenario+1][..., THIS_INTERVAL]).all()

        # (n_tess, n_station)
        self.realres_tess2st_pdch_x = self.res_tess2st_pdch_x[THIS_SCENARIO][..., THIS_INTERVAL]
        assert self.realres_tess2st_pdch_x.shape == (n_tess, n_station)
        # (n_tess, n_interval_window)
        # self.res_tess_pdch_x = self.res_tess2st_pdch_x.sum(axis=1)
        # (n_station, n_interval_window), indicating each station's receiving power
        # at interval t.
        # self.res_station_inp_x = self.res_tess2st_pdch_x.sum(axis=0)

        # (n_tess, n_interval_window), tess's net output power to mg
        # self.res_tess_netoutp_x = self.res_tess_pdch_x - self.res_tess_pch_x
        # (n_station, n_interval_window),
        # indicating each station's net receiving power
        # associated with tess at interval t.
        # self.res_station_netinp_x = self.res_station_inp_x - self.res_station_outp_x

        # tess's energy at interval t, (n_tess, n_interval_window)
        self.res_tess_e_x = [values[model.var_tess_e_x[s_scenario]]
                             for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario-1):
            assert (self.res_tess_e_x[s_scenario][..., THIS_INTERVAL] ==
                   self.res_tess_e_x[s_scenario+1][..., THIS_INTERVAL]).all()

        # (n_tess, )
        self.realres_tess_e_x = self.res_tess_e_x[THIS_SCENARIO][..., THIS_INTERVAL]
        assert self.realres_tess_e_x.shape == (n_tess,)

        # charging sign for tess at interval t, (n_tess, n_interval_window)
        self.res_sign_ch_x = [values[model.var_sign_ch_x[s_scenario]]
                              for s_scenario in range(n_scenario)]

        # for s_scenario in range(n_scenario-1):
        #     try:
        #         assert (self.res_sign_ch_x[s_scenario][..., THIS_INTERVAL] ==
        #            self.res_sign_ch_x[s_scenario+1][..., THIS_INTERVAL]).all()
        #     except:
        #         pass
        #
        # # (n_tess, )
        # self.realres_sign_ch_x = self.res_sign_ch_x[THIS_SCENARIO][..., THIS_INTERVAL]
        # assert self.realres_sign_ch_x.shape == (n_tess, )

        # discharging sign for tess at interval t, (n_tess, n_interval_window)
        # self.res_sign_dch_x = [values[model.var_sign_dch_x[s_scenario]]
        #                       for s_scenario in range(n_scenario)]
        #
        # for s_scenario in range(n_scenario-1):
        #     assert (self.res_sign_dch_x[s_scenario][..., THIS_INTERVAL] ==
        #            self.res_sign_dch_x[s_scenario+1][..., THIS_INTERVAL]).all()
        #
        # # (n_tess, )
        # self.realres_sign_dch_x = self.res_sign_dch_x[THIS_SCENARIO][..., THIS_INTERVAL]
        # assert self.realres_sign_dch_x.shape == (n_tess, )

        # arc status for tess,  (n_tess) list of boolean (n_tsn_arc[i_tess], )
        n_tess = tess_sys.n_tess
        # Important! astype(bool)
        self.res_tess_arc_x = [[values[model.var_tess_arc_x[s_scenario][i_tess]].astype(bool)
                               for i_tess in range(n_tess)]
                               for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario-1):
            for i_tess in range(n_tess):
                assert (tsn_sys.tsn_arc[i_tess][self.res_tess_arc_x[s_scenario][i_tess]][THIS_INTERVAL] ==
                tsn_sys.tsn_arc[i_tess][self.res_tess_arc_x[s_scenario+1][i_tess]][THIS_INTERVAL]).all()

        self.realres_tess_arc_x = [tsn_sys.tsn_arc[i_tess][self.res_tess_arc_x[THIS_SCENARIO][i_tess]][THIS_INTERVAL]
                                   for i_tess in range(n_tess)]
        for i_tess in range(n_tess):
            # arc with 8 columns
            assert self.realres_tess_arc_x[i_tess].shape == (8,)

        # only take the first arc to implement a list (n_tess) of bool array(n_tsn_arc[i_tess])
        # self.res_action_arcs = [tsn_sys.tsn_arc[i_tess][self.res_tess_arc_x[i_tess]][0]
                            # for i_tess in range(n_tess)]
        # self.res_tess_arc_x = values[model.var_tess_arc_x]
        # generate tsn routes from tsn arcs
        # self.gen_tsn_route(tess_sys=tess_sys, tsn_sys=tsn_sys)

        # Transportation status for tess at interval t,  a list (n_scenario) of (n_tess, n_interval_window)
        self.res_sign_onroad_x = [values[model.var_sign_onroad_x[s_scenario]]
                                  for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario-1):
            assert (self.res_sign_onroad_x[s_scenario][..., THIS_INTERVAL] ==
                    self.res_sign_onroad_x[s_scenario+1][..., THIS_INTERVAL]).all()

        self.realres_sign_onroad_x = self.res_sign_onroad_x[THIS_SCENARIO][..., THIS_INTERVAL]
        assert self.realres_sign_onroad_x.shape == (n_tess, )

        # --------------------------------------------------------------------------
        # Station results
        # active power output of station, a list (n_scenario) of (n_station, n_interval_window)
        self.res_station_p_x = [values[model.var_station_p_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario-1):
            assert (self.res_station_p_x[s_scenario][..., THIS_INTERVAL] ==
                    self.res_station_p_x[s_scenario+1][..., THIS_INTERVAL]).all()

        self.realres_station_p_x = self.res_station_p_x[THIS_SCENARIO][..., THIS_INTERVAL]
        assert self.realres_station_p_x.shape == (n_station, )

        # reactive power output of station, a list (n_scenario) of (n_station, n_interval_window)
        self.res_station_q_x = [values[model.var_station_q_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario - 1):
            assert (self.res_station_q_x[s_scenario][..., THIS_INTERVAL] ==
                    self.res_station_q_x[s_scenario + 1][..., THIS_INTERVAL]).all()

        self.realres_station_q_x = self.res_station_q_x[THIS_SCENARIO][
            ..., THIS_INTERVAL]
        assert self.realres_station_q_x.shape == (n_station,)

        # the amount of energy of station, a list (n_scenario) of (n_station, n_interval_window)
        self.res_station_e_x = [values[model.var_station_e_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario - 1):
            assert (self.res_station_e_x[s_scenario][..., THIS_INTERVAL] ==
                    self.res_station_e_x[s_scenario + 1][..., THIS_INTERVAL]).all()

        self.realres_station_e_x = self.res_station_e_x[THIS_SCENARIO][
            ..., THIS_INTERVAL]
        assert self.realres_station_e_x.shape == (n_station,)

        # --------------------------------------------------------------------------
        # distribution system results
        # line active power, a list (n_scenario) of (n_line, n_interval_window)
        self.res_pij_x = [values[model.var_pij_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        # for s_scenario in range(n_scenario - 1):
        #     assert (self.res_pij_x[s_scenario][..., THIS_INTERVAL] ==
        #             self.res_pij_x[s_scenario + 1][..., THIS_INTERVAL]).all()
        #
        # self.realres_pij_x = self.res_pij_x[THIS_SCENARIO][
        #     ..., THIS_INTERVAL]
        # assert self.realres_pij_x.shape == (n_station,)

        # line reactive power, a list (n_scenario) of (n_line, n_interval_window)
        self.res_qij_x = [values[model.var_qij_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        # for s_scenario in range(n_scenario - 1):
        #     assert (self.res_qij_x[s_scenario][..., THIS_INTERVAL] ==
        #             self.res_qij_x[s_scenario + 1][..., THIS_INTERVAL]).all()
        #
        # self.realres_qij_x = self.res_qij_x[THIS_SCENARIO][
        #     ..., THIS_INTERVAL]
        # assert self.realres_qij_x.shape == (n_line,)

        # bus voltage, a list (n_scenario) of (n_bus, n_interval_window)
        self.res_vm_x = [values[model.var_vm_x[s_scenario]]
                         for s_scenario in range(n_scenario)]
        # aggregated active power generation at DS level,
        # a list (n_scenario) of (n_station, n_interval_window)
        self.res_aggregate_pg_x = [values[model.var_aggregate_pg_x[s_scenario]]
                                   for s_scenario in range(n_scenario)]
        # aggregated reactive power generation at DS level,
        # a list (n_scenario) of (n_station, n_interval_window)
        self.res_aggregate_qg_x = [values[model.var_aggregate_qg_x[s_scenario]]
                                   for s_scenario in range(n_scenario)]
        # # sign for load restoration, (n_load, n_interval_window)
        # self.res_load_x = values[self.var_gama_load_x]
        # load restoration
        # a list (n_scenario) of (n_load, n_interval_window)
        n_load = ds_sys.n_load
        self.res_pd_x = [values[model.var_pd_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario - 1):
            try:
                assert (self.res_pd_x[s_scenario][..., THIS_INTERVAL] ==
                    self.res_pd_x[s_scenario + 1][..., THIS_INTERVAL]).all()
            except:
                pass


        self.realres_pd_x = self.res_pd_x[THIS_SCENARIO][
            ..., THIS_INTERVAL]
        assert self.realres_pd_x.shape == (n_load,)

        # a list (n_scenario) of (n_load, n_interval_window)
        self.res_qd_x = [values[model.var_qd_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario - 1):
            assert (self.res_qd_x[s_scenario][..., THIS_INTERVAL] ==
                    self.res_qd_x[s_scenario + 1][..., THIS_INTERVAL]).all()

        self.realres_qd_x = self.res_qd_x[THIS_SCENARIO][
            ..., THIS_INTERVAL]
        assert self.realres_qd_x.shape == (n_load,)


        # # (n_load, n_interval_window)
        # interval = time_sys.interval
        # self.res_pdcut_x = ds_sys.pload[:, interval] - self.res_pd_x
        # # (n_load, n_interval_window)
        # self.res_qdcut_x = ds_sys.qload[:, interval] - self.res_qd_x
        # # a list (n_scenario) of Line connection status, (n_line)
        n_line = ds_sys.n_line

        self.res_alpha_branch_x = [values[model.var_alpha_branch_x[s_scenario]]
                                for s_scenario in range(n_scenario)]

        for s_scenario in range(n_scenario - 1):
            assert (self.res_alpha_branch_x[s_scenario] ==
                    self.res_alpha_branch_x[s_scenario + 1]).all()

        self.realres_alpha_branch_x = self.res_alpha_branch_x[THIS_SCENARIO]
        assert self.realres_alpha_branch_x.shape == (n_line,)

        # # Auxiliary variables for line status,  (n_line)
        # self.res_betaij_x = values[model.var_betaij_x]
        # self.res_betaji_x = values[model.var_betaji_x]

    def consolidate_result_list(self, result_list, time_sys, tc_sys, ds_sys, ss_sys, ts_sys, tess_sys, tsn_sys,
                           if_draw=False):
        '''
        Only for final results consolidation
        :param result_list:
        :param ds_sys:
        :return:
        '''

        '''
            Consolidate the rolling results and unit conversion
            :param result_list: store optimization results
            :param ds_sys: store distribution system parameters
            :return:
            '''

        import numpy as np

        assert isinstance(result_list, list), "the result_list is not a list, please check it"

        ############################################################################
        # Initialization of result arrays
        # --------------------------------------------------------------------------
        # charging power from station to tess at time span t,
        # (n_tess, n_station, n_interval_window)
        n_tess = tess_sys.n_tess
        n_station = ss_sys.n_station
        n_interval = time_sys.n_interval
        # the number of nodes in transportation network
        n_node = ts_sys.n_node

        self.res_tess2st_pch_x = np.zeros((n_tess, n_station, n_interval))

        # discharging power from tess to station at time span t,
        # (n_tess, n_station, n_interval_window)
        self.res_tess2st_pdch_x = np.zeros((n_tess, n_station, n_interval))

        # tess's energy at interval t, (n_tess, n_interval_window)
        self.res_tess_e_x = np.zeros((n_tess, n_interval))

        # charging sign for tess at interval t, (n_tess, n_interval)
        self.res_sign_ch_x = np.zeros((n_tess, n_interval))
        # discharging sign for tess at interval t, (n_tess, n_interval)
        self.res_sign_dch_x = np.zeros((n_tess, n_interval))

        # arc status for tess at interval t,  (n_tess) list of (n_tsn_arc,)
        n_tsn_arc = tsn_sys.n_tsn_arc
        self.res_tess_arc_x = [np.zeros(n_tsn_arc[i_tess], dtype=bool)
                               for i_tess in range(n_tess)]

        # current_location for tess
        self.res_tess_current_location_x = np.zeros((n_tess, n_interval), dtype=object)

        # tess_arcs
        self.res_tess_arcs_x = [np.zeros((0, 8), dtype=int) for i_tess in
                                range(tess_sys.n_tess)]

        # tess_path
        # self.res_tess_path_x = []

        # self.res_tess_arc_x = np.zeros((n_tess, n_tsn_arc), dtype=bool)

        # Transportation status for tess at interval t,  (n_tess, n_interval)
        self.res_sign_onroad_x = np.zeros((n_tess, n_interval))

        # --------------------------------------------------------------------------
        # Station results
        # active power output of station, (n_station, n_interval)
        self.res_station_p_x = np.zeros((n_station, n_interval))
        # reactive power output of station, (n_station, n_interval)
        self.res_station_q_x = np.zeros((n_station, n_interval))
        # the amount of energy of station, (n_station, n_interval)
        self.res_station_e_x = np.zeros((n_station, n_interval))
        # aggregated active power generation at DS level, (n_station, n_interval)
        self.res_aggregate_pg_x = np.zeros((n_station, n_interval))
        # aggregated reactive power generation at DS level, (n_station, n_interval)
        self.res_aggregate_qg_x = np.zeros((n_station, n_interval))

        # --------------------------------------------------------------------------
        # distribution system results
        # line active power, (n_line, n_interval)
        n_line = ds_sys.n_line
        self.res_pij_x = np.zeros((n_line, n_interval))
        # line reactive power, (n_line, n_interval)
        self.res_qij_x = np.zeros((n_line, n_interval))
        # bus voltage, (n_bus, n_interval)
        n_bus = ds_sys.n_bus
        self.res_vm_x = np.zeros((n_bus, n_interval))
        # load restoration
        n_load = ds_sys.n_load
        self.res_pd_x = np.zeros((n_load, n_interval))  # (n_load, n_interval)
        self.res_qd_x = np.zeros((n_load, n_interval))  # (n_load, n_interval)
        # line connection status, (n_line, n_interval)
        self.res_alpha_branch_x = np.zeros((n_line, n_interval))
        # Auxiliary variable for line status, (n_line, n_interval)
        self.res_betaij_x = np.zeros((n_line, n_interval))
        self.res_betaji_x = np.zeros((n_line, n_interval))

        ############################################################################
        # For unit conversion from p.u. to kw, kvar, etc.
        sn_kva = ds_sys.ppnet.sn_kva
        tsn_cut_set = tsn_sys.tsn_cut_set
        # values assignment
        for j_interval in range(time_sys.n_interval):
            self.res_tess2st_pch_x[:, :, j_interval] = result_list[
                                                           j_interval].res_tess2st_pch_x[
                                                       :, :, 0] * sn_kva

            self.res_tess2st_pdch_x[:, :, j_interval] = result_list[
                                                            j_interval].res_tess2st_pdch_x[
                                                        :, :, 0] * sn_kva

            self.res_tess_e_x[:, j_interval] = result_list[
                                                   j_interval].res_tess_e_x[:,
                                               0] * sn_kva

            self.res_sign_ch_x[:, j_interval] = result_list[
                                                    j_interval].res_sign_ch_x[:,
                                                0]

            self.res_sign_dch_x[:, j_interval] = result_list[
                                                     j_interval].res_sign_dch_x[
                                                 :, 0]













            self.res_sign_onroad_x[:, j_interval] = result_list[
                                                        j_interval].res_sign_onroad_x[
                                                    :, 0]

            # ----------------------------------------------------------------------
            self.res_station_p_x[:, j_interval] = result_list[
                                                      j_interval].res_station_p_x[
                                                  :, 0] * sn_kva
            self.res_station_q_x[:, j_interval] = result_list[
                                                      j_interval].res_station_q_x[
                                                  :, 0] * sn_kva
            self.res_station_e_x[:, j_interval] = result_list[
                                                      j_interval].res_station_e_x[
                                                  :, 0] * sn_kva
            self.res_aggregate_pg_x[:, j_interval] = result_list[
                                                         j_interval].res_aggregate_pg_x[
                                                     :, 0] * sn_kva
            self.res_aggregate_qg_x[:, j_interval] = result_list[
                                                         j_interval].res_aggregate_qg_x[
                                                     :, 0] * sn_kva

            # ----------------------------------------------------------------------
            self.res_pij_x[:, j_interval] = result_list[
                                                j_interval].res_pij_x[:,
                                            0] * sn_kva

            self.res_qij_x[:, j_interval] = result_list[
                                                j_interval].res_qij_x[:,
                                            0] * sn_kva

            self.res_vm_x[:, j_interval] = result_list[j_interval].res_vm_x[:,
                                           0]

            self.res_pd_x[:, j_interval] = result_list[
                                               j_interval].res_pd_x[:,
                                           0] * sn_kva

            self.res_qd_x[:, j_interval] = result_list[
                                               j_interval].res_qd_x[:,
                                           0] * sn_kva

            self.res_alpha_branch_x[:, j_interval] = result_list[
                                                         j_interval].res_alpha_branch_x[
                                                     :]

            self.res_betaij_x[:, j_interval] = result_list[
                                                   j_interval].res_betaij_x[:]

            self.res_betaji_x[:, j_interval] = result_list[
                                                   j_interval].res_betaji_x[:]

        # ----------------------------------------------------------------------
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        #
        plt.figure(figsize=(8, 6), dpi=60)

        S_NODE_L = tsn_sys.S_NODE_L
        T_NODE_L = tsn_sys.T_NODE_L
        ARC_LEN = tsn_sys.ARC_LEN

        for i_tess in range(tess_sys.n_tess):
            for j_interval in range(time_sys.n_interval):

                # action_arc
                if not isinstance(
                        result_list[j_interval].current_location[i_tess], str):
                    self.res_tess_current_location_x[i_tess, j_interval] = \
                        result_list[j_interval].current_location[i_tess]
                else:
                    #
                    self.res_tess_current_location_x[i_tess, j_interval] = 'moving node'

                self.res_tess_arcs_x[i_tess] = np.vstack(
                    (self.res_tess_arcs_x[i_tess],
                     result_list[j_interval].res_action_arcs[i_tess]))

                s_node = result_list[j_interval].res_action_arcs[i_tess][
                    S_NODE_L]
                t_node = result_list[j_interval].res_action_arcs[i_tess][
                    T_NODE_L]
                arc_length = result_list[j_interval].res_action_arcs[i_tess][
                    ARC_LEN]

                # why path could be zero? ... This has been corrected in find_travel_time
                path = result_list[j_interval].shortest_path[i_tess][s_node, t_node]
                # path_length = result_list[j_interval].shortest_path_length[i_tess][s_node, t_node]
                # if isinstance(path, list):
                current_node = path[0]
                destination_node = path[-1]
                route_edge_list = [(path[i], path[i+1]) for i in range(len(path)-1)]

                # else:
                #     route_edge_list = []
                #     current_node = self.res_tess_current_location_x[i_tess, j_interval]
                plt.cla()

                # Set title
                plt.title("tess_{}_interval_{}".format(i_tess, j_interval))
                # plt.grid(True)
                plt.axis('off')

                pos = tc_sys.tsc['node_position']
                if 'moving node' in result_list[j_interval].tess_graph[i_tess].nodes:
                    u, v = result_list[j_interval].tess_graph[i_tess]['moving node']
                    distance2u = result_list[j_interval].tess_graph[i_tess].edges['moving node', u]['length']
                    distance2v = result_list[j_interval].tess_graph[i_tess].edges['moving node', v]['length']
                    u_x, u_y = pos[u]
                    v_x, v_y = pos[v]
                    moving_x = (distance2u * v_x + distance2v * u_x) / (distance2u + distance2v)
                    moving_y = (distance2u * v_y + distance2v * u_y) / (distance2u + distance2v)
                    pos['moving node'] = np.array([moving_x, moving_y])

                # draw the base map
                nx.draw_networkx(result_list[j_interval].tess_graph[i_tess],
                    pos, nodelist=range(n_node), node_color='r', edge_color='k',
                    alpha=0.8)
                # draw the current position
                nx.draw_networkx_nodes(result_list[j_interval].tess_graph[i_tess],
                                 pos, nodelist=[current_node],
                                 node_color='b')
                # draw the destination
                nx.draw_networkx_nodes(result_list[j_interval].tess_graph[i_tess],
                    pos, nodelist=[destination_node],
                    node_color='c')

                # highlight the route from current position to destination
                nx.draw_networkx_edges(
                    result_list[j_interval].tess_graph[i_tess], pos,
                    edgelist=route_edge_list,
                    edge_color='b', width=10, alpha=0.5)

                plt.savefig('temp_figure/tess_{}_interval_{}.png'.format(i_tess, j_interval))

                pass

        ############################################################################
        # append the initial energy of staion
        self.res_station_e_x = np.hstack(([ss_sys.station_e_u * sn_kva,
                                           self.res_station_e_x]))
        # append the initial energy of tess
        self.res_tess_e_x = np.hstack([tess_sys.tess_e_init * sn_kva,
                                       self.res_tess_e_x])

        # soc of tess
        self.res_tess_soc_x = self.res_tess_e_x / (
                tess_sys.tess_cap_e * sn_kva) * 100

        # induce additional results from base results
        # (n_tess, n_interval)
        self.res_tess_pch_x = self.res_tess2st_pch_x.sum(axis=1)
        # (n_station, n_interval, indicating each station's sending power at
        # interval t
        self.res_station_outp_x = self.res_tess2st_pch_x.sum(axis=0)

        # (n_tess, n_interval)
        self.res_tess_pdch_x = self.res_tess2st_pdch_x.sum(axis=1)
        # (n_station, n_interval), indicating each station's receiving power
        # at interval t.
        self.res_station_inp_x = self.res_tess2st_pdch_x.sum(axis=0)

        # (n_tess, n_interval), tess's net output power to mg
        self.res_tess_netoutp_x = self.res_tess_pdch_x - self.res_tess_pch_x
        # (n_station, n_interval), indicating each station's net receiving power
        # associated with tess at interval t.
        self.res_station_netinp_x = self.res_station_inp_x - self.res_station_outp_x

        # (n_load, n_interval)
        self.res_pdcut_x = ds_sys.pload * sn_kva - self.res_pd_x
        # (n_load, n_interval)
        self.res_qdcut_x = ds_sys.qload * sn_kva - self.res_qd_x

        # generate tsn_routes from tsn arcs
        # self.res_tess_route
        self.gen_tsn_route(tess_sys=tess_sys, tsn_sys=tsn_sys)

        ############################################################################
        # Calculate objective value
        delta_t = time_sys.delta_t

        # transportation cost of tess
        tess_cost_transportation = np.tile(tess_sys.tess_cost_transportation,
                                           reps=(1, n_interval))
        self.res_cost_tess_transportation = (self.res_sign_onroad_x *
                                             tess_cost_transportation).sum()

        # charging/discharging cost
        tess_cost_power = np.tile(tess_sys.tess_cost_power, reps=(1, n_interval))
        self.res_cost_tess_power = ((self.res_tess_pch_x + self.res_tess_pdch_x)
                                    * tess_cost_power * delta_t).sum()

        # station generation cost
        station_gencost = np.tile(ss_sys.station_gencost, reps=(1, n_interval))
        self.res_cost_gencost = (station_gencost * self.res_station_p_x
                                 * delta_t).sum()

        # load interruption cost
        load_interruption_cost = np.tile(ds_sys.load_interruption_cost,
                                         reps=(1, n_interval))
        self.res_cost_load_interruption = (
                (ds_sys.pload * sn_kva - self.res_pd_x) *
                load_interruption_cost * delta_t).sum()

        # the total cost
        self.res_cost_total = self.res_cost_tess_transportation + \
                              self.res_cost_tess_power \
                              + self.res_cost_gencost + self.res_cost_load_interruption

        # Load restoration
        idx_critical_load = np.flatnonzero(ds_sys.ppnet.load['load_cost'] >= 5)
        idx_normal_load = np.flatnonzero(ds_sys.ppnet.load['load_cost'] < 5)

        # restoration rate for critical loads (%)
        self.res_restoration_rate_critical = self.res_pd_x[
                                             idx_critical_load, :].sum() / (
                                                     ds_sys.pload[
                                                         idx_critical_load,
                                                         :].sum()
                                                     * sn_kva)
        # restoration rate for normal loads (%)
        self.res_restoration_rate_normal = self.res_pd_x[
                                           idx_normal_load, :].sum() / (
                                                   ds_sys.pload[
                                                       idx_normal_load, :].sum()
                                                   * sn_kva)

        # retoration rate for all loads (%)
        self.res_restoration_rate = self.res_pd_x.sum() / (ds_sys.pload.sum()
                                                           * sn_kva)

        # Calculate load restoration in each island
        self.res_pd_island = np.zeros((ss_sys.n_microgrid, n_interval))

        for i_microgrid in range(ss_sys.n_microgrid):
            # 32 is the # of load in each distribution system
            idx_load_island = np.arange(32) + i_microgrid * 32
            self.res_pd_island[i_microgrid, :] = self.res_pd_x[
                                                 idx_load_island, :].sum(axis=0)

        # Calculate energy transfer among microgrids
        n_timepoint = n_interval + 1
        self.res_e_transfer_station = np.zeros((n_station, n_timepoint))

        for j_interval in range(n_interval):
            self.res_e_transfer_station[:, j_interval + 1] = \
                self.res_e_transfer_station[:, j_interval] + \
                self.res_station_netinp_x[:, j_interval] * delta_t

        # Find the topology
        self.res_switch_off_line = []

        for j_interval in range(n_interval):
            idx_switch_off_line = np.flatnonzero(
                self.res_alpha_branch_x[:, j_interval] == 0)

            self.res_switch_off_line.append(ds_sys.ppnet.line.loc[
                                            idx_switch_off_line,
                                            'from_bus':'to_bus'])

        # draw figures
        if if_draw:
            from pytess.utils import draw_figure
            draw_figure(result=self, ds_sys=ds_sys, ss_sys=ss_sys, tess_sys=tess_sys,
                        tsn_sys=tsn_sys)

    def consolidate_result_list_2stage(self, result_list, time_sys, scenario_sys,
        tc_sys, ds_sys, ss_sys, ts_sys, tess_sys, tsn_sys, if_draw=False):
        '''
        Only for final results consolidation
        :param result_list:
        :param ds_sys:
        :return:
        '''

        '''
            Consolidate the rolling results and unit conversion
            :param result_list: store optimization results
            :param ds_sys: store distribution system parameters
            :return:
            '''

        import numpy as np

        assert isinstance(result_list, list), "the result_list is not a list, please check it"

        ############################################################################
        # Initialization of result arrays
        # --------------------------------------------------------------------------
        # charging power from station to tess at time span t,
        # (n_tess, n_station, n_interval_window)
        n_tess = tess_sys.n_tess
        n_station = ss_sys.n_station
        n_interval = time_sys.n_interval
        # the number of nodes in transportation network
        n_node = ts_sys.n_node

        self.finalres_tess2st_pch_x = np.zeros((n_tess, n_station, n_interval))

        # discharging power from tess to station at time span t,
        # (n_tess, n_station, n_interval_window)
        self.finalres_tess2st_pdch_x = np.zeros((n_tess, n_station, n_interval))

        # tess's energy at interval t, (n_tess, n_interval_window)
        self.finalres_tess_e_x = np.zeros((n_tess, n_interval))

        # charging sign for tess at interval t, (n_tess, n_interval)
        self.finalres_sign_ch_x = np.zeros((n_tess, n_interval))
        # discharging sign for tess at interval t, (n_tess, n_interval)
        self.finalres_sign_dch_x = np.zeros((n_tess, n_interval))

        # arc status for tess at interval t,  (n_tess) list of (n_tsn_arc,)
        n_tsn_arc = tsn_sys.n_tsn_arc
        self.finalres_tess_arc_x = [np.zeros(n_tsn_arc[i_tess], dtype=bool)
                               for i_tess in range(n_tess)]

        # current_location for tess
        n_timepoint = time_sys.n_timepoint
        self.finalres_tess_current_location_x = np.zeros((n_tess, n_timepoint), dtype=object)

        # tess_arcs
        self.finalres_tess_arcs_x = [np.zeros((0, 8), dtype=int) for i_tess in
                                range(tess_sys.n_tess)]

        # tess_path
        # self.res_tess_path_x = []

        # self.res_tess_arc_x = np.zeros((n_tess, n_tsn_arc), dtype=bool)

        # Transportation status for tess at interval t,  (n_tess, n_interval)
        self.finalres_sign_onroad_x = np.zeros((n_tess, n_interval))

        # --------------------------------------------------------------------------
        # Station results
        # active power output of station, (n_station, n_interval)
        self.finalres_station_p_x = np.zeros((n_station, n_interval))
        # reactive power output of station, (n_station, n_interval)
        self.finalres_station_q_x = np.zeros((n_station, n_interval))
        # the amount of energy of station, (n_station, n_interval)
        self.finalres_station_e_x = np.zeros((n_station, n_interval))
        # aggregated active power generation at DS level, (n_station, n_interval)
        self.finalres_aggregate_pg_x = np.zeros((n_station, n_interval))
        # aggregated reactive power generation at DS level, (n_station, n_interval)
        self.finalres_aggregate_qg_x = np.zeros((n_station, n_interval))

        # --------------------------------------------------------------------------
        # distribution system results
        # line active power, (n_line, n_interval)
        n_line = ds_sys.n_line
        self.finalres_pij_x = np.zeros((n_line, n_interval))
        # line reactive power, (n_line, n_interval)
        self.finalres_qij_x = np.zeros((n_line, n_interval))
        # bus voltage, (n_bus, n_interval)
        n_bus = ds_sys.n_bus
        self.finalres_vm_x = np.zeros((n_bus, n_interval))
        # load restoration
        n_load = ds_sys.n_load
        self.finalres_pd_x = np.zeros((n_load, n_interval))  # (n_load, n_interval)
        self.finalres_qd_x = np.zeros((n_load, n_interval))  # (n_load, n_interval)
        # line connection status, (n_line, n_interval)
        self.finalres_alpha_branch_x = np.zeros((n_line, n_interval))
        # # Auxiliary variable for line status, (n_line, n_interval)
        # self.res_betaij_x = np.zeros((n_line, n_interval))
        # self.res_betaji_x = np.zeros((n_line, n_interval))

        ############################################################################
        # For unit conversion from p.u. to kw, kvar, etc.
        sn_mva = ds_sys.ppnet.sn_mva
        tsn_cut_set = tsn_sys.tsn_cut_set
        # values assignment
        for j_interval in range(time_sys.n_interval):
            self.finalres_tess2st_pch_x[:, :, j_interval] = \
                result_list[j_interval].realres_tess2st_pch_x * sn_mva

            self.finalres_tess2st_pdch_x[:, :, j_interval] = \
                result_list[j_interval].realres_tess2st_pdch_x * sn_mva

            self.finalres_tess_e_x[:, j_interval] = \
                result_list[j_interval].realres_tess_e_x * sn_mva






            self.finalres_sign_onroad_x[:, j_interval] = \
                result_list[j_interval].realres_sign_onroad_x

            # ----------------------------------------------------------------------
            self.finalres_station_p_x[:, j_interval] = \
                result_list[j_interval].realres_station_p_x * sn_mva

            self.finalres_station_q_x[:, j_interval] = \
                result_list[j_interval].realres_station_q_x * sn_mva

            self.finalres_station_e_x[:, j_interval] = \
                result_list[j_interval].realres_station_e_x * sn_mva

            # self.res_aggregate_pg_x[:, j_interval] = result_list[j_interval].res_aggregate_pg_x[
            #                                          :, 0] * sn_mva
            # self.res_aggregate_qg_x[:, j_interval] = result_list[
            #                                              j_interval].res_aggregate_qg_x[
            #                                          :, 0] * sn_mva

            # ----------------------------------------------------------------------
            # self.res_pij_x[:, j_interval] = result_list[
            #                                     j_interval].res_pij_x[:,
            #                                 0] * sn_mva
            #
            # self.res_qij_x[:, j_interval] = result_list[
            #                                     j_interval].res_qij_x[:,
            #                                 0] * sn_mva
            #
            # self.res_vm_x[:, j_interval] = result_list[j_interval].res_vm_x[:,
            #                                0]

            self.finalres_pd_x[:, j_interval] = \
                result_list[j_interval].realres_pd_x * sn_mva

            self.finalres_qd_x[:, j_interval] = \
                result_list[j_interval].realres_qd_x * sn_mva

            self.finalres_alpha_branch_x[:, j_interval] = \
                result_list[j_interval].realres_alpha_branch_x

            # self.res_betaij_x[:, j_interval] = result_list[
            #                                        j_interval].res_betaij_x[:]
            #
            # self.res_betaji_x[:, j_interval] = result_list[
            #                                        j_interval].res_betaji_x[:]

        # ----------------------------------------------------------------------
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        #
        plt.figure(figsize=(8, 6), dpi=60)

        S_NODE_L = tsn_sys.S_NODE_L
        T_NODE_L = tsn_sys.T_NODE_L
        ARC_LEN = tsn_sys.ARC_LEN

        for i_tess in range(tess_sys.n_tess):
            for j_interval in range(time_sys.n_interval):

                # action_arc
                if not isinstance(
                        result_list[j_interval].current_location[i_tess], str):
                    self.finalres_tess_current_location_x[i_tess, j_interval] = \
                        result_list[j_interval].current_location[i_tess]
                else:
                    #
                    self.finalres_tess_current_location_x[i_tess, j_interval] = 'moving node'

                self.finalres_tess_arcs_x[i_tess] = np.vstack(
                    (self.finalres_tess_arcs_x[i_tess],
                     result_list[j_interval].realres_tess_arc_x[i_tess]))

                if j_interval == time_sys.n_interval-1:
                    # if the last inberval, need to append the destination,
                    # arcs are indicated in index of station, not the real node_i,
                    # so it needs station_node
                    station_node = ss_sys.station_node
                    self.finalres_tess_current_location_x[i_tess, j_interval+1] \
                        = station_node[result_list[j_interval].realres_tess_arc_x[i_tess][T_NODE_L]]

                s_node = result_list[j_interval].realres_tess_arc_x[i_tess][
                    S_NODE_L]
                t_node = result_list[j_interval].realres_tess_arc_x[i_tess][
                    T_NODE_L]
                arc_length = result_list[j_interval].realres_tess_arc_x[i_tess][
                    ARC_LEN]

                # why path could be zero? ... This has been corrected in find_travel_time
                path = result_list[j_interval].shortest_path[i_tess][s_node, t_node]
                # path_length = result_list[j_interval].shortest_path_length[i_tess][s_node, t_node]
                # if isinstance(path, list):
                current_node = path[0]
                destination_node = path[-1]
                route_edge_list = [(path[i], path[i+1]) for i in range(len(path)-1)]

                # else:
                #     route_edge_list = []
                #     current_node = self.res_tess_current_location_x[i_tess, j_interval]
                plt.cla()

                # Set title
                plt.title("tess_{}_interval_{}".format(i_tess, j_interval))
                # plt.grid(True)
                plt.axis('off')

                pos = tc_sys.tsc['node_position']
                if 'moving node' in result_list[j_interval].tess_graph[i_tess].nodes:
                    u, v = result_list[j_interval].tess_graph[i_tess]['moving node']
                    distance2u = result_list[j_interval].tess_graph[i_tess].edges['moving node', u]['length']
                    distance2v = result_list[j_interval].tess_graph[i_tess].edges['moving node', v]['length']
                    u_x, u_y = pos[u]
                    v_x, v_y = pos[v]
                    moving_x = (distance2u * v_x + distance2v * u_x) / (distance2u + distance2v)
                    moving_y = (distance2u * v_y + distance2v * u_y) / (distance2u + distance2v)
                    pos['moving node'] = np.array([moving_x, moving_y])

                # draw the base map
                nx.draw_networkx(result_list[j_interval].tess_graph[i_tess],
                    pos, nodelist=range(n_node), node_color='r', edge_color='k',
                    alpha=0.8)
                # draw the current position
                nx.draw_networkx_nodes(result_list[j_interval].tess_graph[i_tess],
                                 pos, nodelist=[current_node],
                                 node_color='b')
                # draw the destination
                nx.draw_networkx_nodes(result_list[j_interval].tess_graph[i_tess],
                    pos, nodelist=[destination_node],
                    node_color='c')

                # highlight the route from current position to destination
                nx.draw_networkx_edges(
                    result_list[j_interval].tess_graph[i_tess], pos,
                    edgelist=route_edge_list,
                    edge_color='b', width=10, alpha=0.5)

                # time_sys.time_stamp
                directory_name = 'results/moving_figure/' + time_sys.time_stamp

                # import os
                # if not os.path.exists(directory_name):
                #     # multi-layer directory, makedirs, otherwise mkdir is also available
                #     os.makedirs(directory_name)

                # pathlib is preferred to os.path
                import pathlib
                # exist_ok=Ture means there is no error raised up when the directory already exists.
                # otherwise, it will raise errors.
                pathlib.Path(directory_name).mkdir(parents=True, exist_ok=True)

                plt.savefig(directory_name + '/tess_{i_tess}_interval_{j_interval}.png'
                            .format(i_tess=i_tess, j_interval=j_interval))

                pass

        # generate tsn route from self.finalres_tess_current_location_x(n_tess, m_interval)
        self.gen_tsn_route_2stage(time_sys=time_sys, ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys)

        ############################################################################
        # append the initial energy of staion
        self.finalres_station_e_x = np.hstack(([ss_sys.station_e_u * sn_mva,
                                           self.finalres_station_e_x]))
        # append the initial energy of tess
        self.finalres_tess_e_x = np.hstack([tess_sys.tess_e_init * sn_mva,
                                       self.finalres_tess_e_x])

        # soc of tess
        self.finalres_tess_soc_x = self.finalres_tess_e_x / (
                tess_sys.tess_cap_e * sn_mva) * 100

        # induce additional results from base results
        # (n_tess, n_interval)
        self.finalres_tess_pch_x = self.finalres_tess2st_pch_x.sum(axis=1)
        # (n_station, n_interval, indicating each station's sending power at
        # interval t
        self.finalres_station_outp_x = self.finalres_tess2st_pch_x.sum(axis=0)

        # (n_tess, n_interval)
        self.finalres_tess_pdch_x = self.finalres_tess2st_pdch_x.sum(axis=1)
        # (n_station, n_interval), indicating each station's receiving power
        # at interval t.
        self.finalres_station_inp_x = self.finalres_tess2st_pdch_x.sum(axis=0)

        # (n_tess, n_interval), tess's net output power to mg
        self.finalres_tess_netoutp_x = self.finalres_tess_pdch_x - self.finalres_tess_pch_x
        # (n_station, n_interval), indicating each station's net receiving power
        # associated with tess at interval t.
        self.finalres_station_netinp_x = self.finalres_station_inp_x - self.finalres_station_outp_x

        # (n_load, n_interval)
        self.finalres_pdcut_x = ds_sys.pload * sn_mva - self.finalres_pd_x
        # (n_load, n_interval)
        self.finalres_qdcut_x = ds_sys.qload * sn_mva - self.finalres_qd_x

        # generate tsn_routes from tsn arcs
        # self.res_tess_route
        # self.gen_tsn_route(tess_sys=tess_sys, tsn_sys=tsn_sys)

        ############################################################################
        # Calculate objective value
        delta_t = time_sys.delta_t

        # transportation cost of tess
        tess_cost_transportation = np.tile(tess_sys.tess_cost_transportation,
                                           reps=(1, n_interval))
        self.finalres_cost_tess_transportation = (self.finalres_sign_onroad_x *
                                             tess_cost_transportation).sum()

        # charging/discharging cost
        tess_cost_power = np.tile(tess_sys.tess_cost_power, reps=(1, n_interval))
        self.finalres_cost_tess_power = ((self.finalres_tess_pch_x + self.finalres_tess_pdch_x)
                                    * tess_cost_power * delta_t).sum()

        # station generation cost
        station_gencost = np.tile(ss_sys.station_gencost, reps=(1, n_interval))
        self.finalres_cost_gencost = (station_gencost * self.finalres_station_p_x
                                 * delta_t).sum()

        # load interruption cost
        load_interruption_cost = np.tile(ds_sys.load_interruption_cost,
                                         reps=(1, n_interval))
        self.finalres_cost_load_interruption = (
                (ds_sys.pload * sn_mva - self.finalres_pd_x) *
                load_interruption_cost * delta_t).sum()

        # the total cost
        self.finalres_cost_total = self.finalres_cost_tess_transportation + \
                              self.finalres_cost_tess_power \
                              + self.finalres_cost_gencost + self.finalres_cost_load_interruption

        # Load restoration
        idx_critical_load = np.flatnonzero(ds_sys.ppnet.load['load_cost'] >= 5000)
        idx_normal_load = np.flatnonzero(ds_sys.ppnet.load['load_cost'] < 5000)

        # restoration rate for critical loads (%)
        self.finalres_restoration_rate_critical = self.finalres_pd_x[
                                             idx_critical_load, :].sum() / (
                                                     ds_sys.pload[
                                                         idx_critical_load,
                                                         :].sum()
                                                     * sn_mva)
        # restoration rate for normal loads (%)
        self.finalres_restoration_rate_normal = self.finalres_pd_x[
                                           idx_normal_load, :].sum() / (
                                                   ds_sys.pload[
                                                       idx_normal_load, :].sum()
                                                   * sn_mva)

        # retoration rate for all loads (%)
        self.finalres_restoration_rate = self.finalres_pd_x.sum() / (ds_sys.pload.sum()
                                                           * sn_mva)

        # Calculate load restoration in each island
        self.finalres_pd_island = np.zeros((ss_sys.n_microgrid, n_interval))

        for i_microgrid in range(ss_sys.n_microgrid):
            # 32 is the # of load in each distribution system
            idx_load_island = np.arange(32) + i_microgrid * 32
            self.finalres_pd_island[i_microgrid, :] = self.finalres_pd_x[
                                                 idx_load_island, :].sum(axis=0)

        # Calculate energy transfer among microgrids
        n_timepoint = n_interval + 1
        self.finalres_e_transfer_station = np.zeros((n_station, n_timepoint))

        for j_interval in range(n_interval):
            self.finalres_e_transfer_station[:, j_interval + 1] = \
                self.finalres_e_transfer_station[:, j_interval] + \
                self.finalres_station_netinp_x[:, j_interval] * delta_t

        # Find the topology
        self.finalres_switch_off_line = []

        for j_interval in range(n_interval):
            idx_switch_off_line = np.flatnonzero(
                self.finalres_alpha_branch_x[:, j_interval] == 0)

            self.finalres_switch_off_line.append(ds_sys.ppnet.line.loc[
                                            idx_switch_off_line,
                                            'from_bus':'to_bus'])

        # draw figures
        if if_draw:
            from pytess.utils import draw_figure
            draw_figure(result=self, ds_sys=ds_sys, ss_sys=ss_sys, tess_sys=tess_sys,
                        tsn_sys=tsn_sys)

    def gen_tsn_route(self, tess_sys, tsn_sys):

        n_tess = tess_sys.n_tess
        # retrieve tess's routes by sorting sequence of arcs
        # a list of # of TESS, each depicting the tess's routes
        self.res_tess_route_x = []
        # two columns, indicating each tess's active arcs
        # np.nonzero() return and pass tuple to vstack()
        # [i_tess, j_arc]
        # active_arc = np.vstack(np.nonzero(result.res_tess_arc_x[1])).T

        for i_tess in range(n_tess):
            active_arc_temp = np.flatnonzero(self.res_tess_arc_x[i_tess])

            self.res_tess_route_x.append(
                tsn_sys.tsn_arc[i_tess][active_arc_temp, :])

    def gen_tsn_route_2stage(self, time_sys, ss_sys, tess_sys, tsn_sys):

        n_tess = tess_sys.n_tess
        n_interval = time_sys.n_interval
        n_timepoint = time_sys.n_timepoint
        # array([7, 1, 2, 16, 23])
        station_node = ss_sys.station_node

        self.finalres_tess_route_x  = []

        for i_tess in range(n_tess):
            # set (2, n_timepoint) array
            self.finalres_tess_route_x.append(
                np.vstack([np.arange(n_timepoint),
                           self.finalres_tess_current_location_x[i_tess]]))

            assert self.finalres_tess_route_x[i_tess].shape == (2, n_timepoint)
            delete_column = []

            for j_timepoint in range(n_timepoint):
                current_location = \
                    self.finalres_tess_route_x[i_tess][1, j_timepoint]

                if not isinstance(current_location, str):
                    try:
                        current_location_index = np.flatnonzero(station_node == current_location)
                        if current_location_index.size:
                            self.finalres_tess_route_x[i_tess][1, j_timepoint] = current_location_index.item()
                        else:
                            # current location is not at station
                            delete_column.append(j_timepoint)
                    except:
                        pass
                else:
                    # self.finalres_tess_route_x[i_tess] = np.delete(
                    #   self.finalres_tess_route_x[i_tess], j_timepoint, axis=1)
                    delete_column.append(j_timepoint)

            self.finalres_tess_route_x[i_tess] = np.delete(
                self.finalres_tess_route_x[i_tess], delete_column, axis=1)

        pass


    def write_to_excel(self, time_sys, ds_sys, ss_sys, tess_sys, tsn_sys):
        '''
        Write data to excel for originpro
        :param ds_sys:
        :param ss_sys:
        :param tess_sys:
        :param tsn_sys:
        :return:
        '''

        import pandas as pd
        from utils import multiple_df2single_sheet

        # Initialization of writer

        directory_name = 'results/excel'
        # import os
        # if not os.path.exists(directory_name):
        #     os.mkdir(directory_name)

        # pathlib is preferred to os.path
        import pathlib
        # exist_ok=Ture means there is no error raised up when the directory already exists.
        # otherwise, it will raise errors.
        pathlib.Path(directory_name).mkdir(parents=True, exist_ok=True)

        writer = pd.ExcelWriter(directory_name + '/' + time_sys.time_stamp + '.xlsx',
                                engine='xlsxwriter')
        # ----------------------------------------------------------------------
        # Write MESS_based sheets
        df_list = []
        for i_tess in range(tess_sys.n_tess):
            pass
            df_list.append(
                pd.DataFrame(self.finalres_tess_route_x[i_tess],
                             index=['time point', 'position']))

        df_list.append(
            pd.DataFrame(self.finalres_tess_current_location_x))

        df_list.append(pd.DataFrame(self.finalres_tess_pch_x,
                                    index=['mess {}'.format(i_tess)
                                           for i_tess in range(tess_sys.n_tess)]))

        df_list.append(pd.DataFrame(self.finalres_tess_pdch_x,
                                    index=['mess {}'.format(i_tess)
                                           for i_tess in range(tess_sys.n_tess)]))

        df_list.append(pd.DataFrame(self.finalres_tess_soc_x,
                                    index=['mess {}'.format(i_tess)
                                           for i_tess in range(tess_sys.n_tess)]))

        multiple_df2single_sheet(df_list=df_list, sheet_name='mess',
                                 writer=writer, space=2)

        # ----------------------------------------------------------------------
        # Write Station-based sheets
        df_list = []

        for i_station in range(ss_sys.n_station):
            # (n_tess, n_interval)
            finaltess_p2this_station = (self.finalres_tess2st_pdch_x -
                                   self.finalres_tess2st_pch_x)[:, i_station, :]

            df_list.append(pd.DataFrame(finaltess_p2this_station,
                                        index=['tess {}'.format(i_tess)
                                               for i_tess in
                                               range(tess_sys.n_tess)]))

        df_list.append(pd.DataFrame(self.finalres_station_p_x,
                                    index=['station {}'.format(i_station)
                                           for i_station in
                                           range(ss_sys.n_station)]))

        # df_list.append(pd.DataFrame(self.res_station_e_x,
        #                             index=['station {}'.format(i_station)
        #                                    for i_station in
        #                                    range(ss_sys.n_station)]))

        df_list.append(pd.DataFrame(self.finalres_e_transfer_station,
                                    index=['station {}'.format(i_station)
                                           for i_station in
                                           range(ss_sys.n_station)]))

        df_list.append(pd.DataFrame(self.finalres_pd_island,
                                    index=['island {}'.format(i_microgrid)
                                           for i_microgrid in
                                           range(ss_sys.n_microgrid)]))

        multiple_df2single_sheet(df_list=df_list, sheet_name='island',
                                 writer=writer, space=2)
        # ----------------------------------------------------------------------
        # Save to excel file
        writer.save()