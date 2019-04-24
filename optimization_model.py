import numpy as np
from numpy import array, zeros, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import cplex as cpx

class OptimizationModel(cpx.Cplex):

    def add_variables(self, time_sys, scenario_sys, ds_sys, ss_sys, tess_sys, tsn_sys):

        # from pytess.optimization_model_toolbox import add_variables_cpx
        #
        # add_variables_cpx(self=self, ds_sys=ds_sys, ss_sys=ss_sys,
        #                   tess_sys=tess_sys, tsn_sys=tsn_sys)

        ## Define varialbles and get position array
        # prefix var_ is for variables index array
        # TESS model
        # charging power from station to tess at interval t, 3D-array,
        # (n_tess, n_station, n_interval_window)

        import copy
        n_scenario = scenario_sys.n_scenario

        pload_realization = copy.deepcopy(scenario_sys.pload_realization)
        qload_realization = copy.deepcopy(scenario_sys.qload_realization)
        pload_scenario = copy.deepcopy(scenario_sys.pload_scenario)
        qload_scenario = copy.deepcopy(scenario_sys.qload_scenario)

        n_tess = tess_sys.n_tess
        n_station = ss_sys.n_station
        # the interval is only for this time window
        n_interval_window = time_sys.n_interval_window
        interval = time_sys.interval

        #
        pload_scenario[:, :, interval[0]] = np.tile(pload_realization[:, interval[0]], reps=(n_scenario, 1))
        qload_scenario[:, :, interval[0]] = np.tile(qload_realization[:, interval[0]], reps=(n_scenario, 1))

        # Considering the scenario, all the variables need to be encapsulated in scenario list
        self.var_tess2st_pch_x = [array(self.variables.add(
            names=['ch_tess{0}_st{1}_t{2}'.format(i_tess, j_station, k_interval)
                   for i_tess in range(n_tess)
                   for j_station in range(n_station)
                   for k_interval in range(n_interval_window)])
            ).reshape(n_tess, n_station, n_interval_window)
            for s_scenario in range(n_scenario)]

        # discharging power from tess to station at interval t, 3D-array,
        # (n_tess, n_station, n_interval_window)
        self.var_tess2st_pdch_x = [array(self.variables.add(
            names=[
                'dch_tess{0}_st{1}_t{2}'.format(i_tess, j_station, k_interval)
                for i_tess in range(n_tess)
                for j_station in range(n_station)
                for k_interval in range(n_interval_window)])
            ).reshape(n_tess, n_station, n_interval_window)
            for s_scenario in range(n_scenario)]

        # tess's energy at interval t, (n_tess, n_interval_window)
        # lb should be 1-d array-like input with the same length as variables
        # .flatten() returns copy while ravel() generally returns view.
        tess_e_l = np.tile(tess_sys.tess_e_l, reps=(1, n_interval_window))
        tess_e_u = np.tile(tess_sys.tess_e_u, reps=(1, n_interval_window))

        self.var_tess_e_x = [array(self.variables.add(
            lb=tess_e_l.ravel(), ub=tess_e_u.ravel(),
            names=['e_tess{0}_t{1}'.format(i_tess, j_interval)
                   for i_tess in range(n_tess)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_tess, n_interval_window)
            for s_scenario in range(n_scenario)]

        # charging sign for tess at interval t, (n_tess, n_interval_window)
        self.var_sign_ch_x = [array(self.variables.add(
            types=['B'] * (n_tess * n_interval_window),
            names=['sign_ch_tess{0}_t{1}'.format(i_tess, j_interval)
                   for i_tess in range(n_tess)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_tess, n_interval_window)
            for s_scenario in range(n_scenario)]

        # discharging sign for ev at interval t, (n_tess, n_interval_window)
        self.var_sign_dch_x = [array(self.variables.add(
            types=['B'] * (n_tess * n_interval_window),
            names=['sign_dch_tess{0}_t{1}'.format(i_tess, j_interval)
                   for i_tess in range(n_tess)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_tess, n_interval_window)
            for s_scenario in range(n_scenario)]

        # arc status for tess at interval t, (n_tess, n_tsn_arc)
        # modify it to fit into new tsn model
        # tsn_arc = tsn_sys.tsn_arc[]
        n_tsn_arc = tsn_sys.n_tsn_arc

        # self.var_tess_arc_x = [
        #     self.variables.add(
        #         types=['B'] * n_tess * n_tsn_arc[i_tess],
        #         names=['tess{0}_arc{1}'.format(i_tess, j_arc)
        #            for i_tess in range(n_tess)
        #            for j_arc in range(n_tsn_arc[i_tess])
        #     )
        # ]
        # This is a list since each tess's the no. tsn arc is diffefent,
        # list comprehension to generate a (n_scenario) list of (n_tess) list of array (n_tsn_arc[i_tess], )
        self.var_tess_arc_x = [[array(self.variables.add(
            types=['B'] * n_tsn_arc[i_tess],
            names=['tess{0}_arc{1}'.format(i_tess, j_arc)
                   for j_arc in range(n_tsn_arc[i_tess])]))
            for i_tess in range(n_tess)]
            for s_scenario in range(n_scenario)]

        # Transit status for tess at time span t, (n_tess, n_interval_window)
        self.var_sign_onroad_x = [array(self.variables.add(
            types=['B'] * (n_tess * n_interval_window),
            names=['sign_onroad_tess{0}_t{1}'.format(i_tess, j_interval)
                   for i_tess in range(n_tess)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_tess, n_interval_window)
            for s_scenario in range(n_scenario)]

        ## MG model
        # active power output of station, (n_station, n_interval_window)
        station_p_l = np.tile(ss_sys.station_p_l, reps=(1, n_interval_window))
        station_p_u = np.tile(ss_sys.station_p_u, reps=(1, n_interval_window))
        p_localload = ss_sys.p_localload[:, interval]

        # the local load uncertainties are not considered.
        self.var_station_p_x = [array(self.variables.add(
            lb=station_p_l.ravel(),
            ub=(station_p_u - p_localload).ravel(),
            names=['p_station{0}_t{1}'.format(i_station, j_interval)
                   for i_station in range(n_station)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_station, n_interval_window)
            for s_scenario in range(n_scenario)]

        # reactive power output of station, (n_station, n_interval_window)
        station_q_l = np.tile(ss_sys.station_q_l, reps=(1, n_interval_window))
        station_q_u = np.tile(ss_sys.station_q_u, reps=(1, n_interval_window))
        q_localload = ss_sys.q_localload[:, interval]

        self.var_station_q_x = [array(self.variables.add(
            lb=(station_q_l - q_localload).ravel(),
            ub=(station_q_u - q_localload).ravel(),
            names=['q_station{0}_t{1}'.format(i_station, j_interval)
                   for i_station in range(n_station)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_station, n_interval_window)
            for s_scenario in range(n_scenario)]

        # the amount of energy of station, (n_station, n_interval_window)
        station_e_l = np.tile(ss_sys.station_e_l, reps=(1, n_interval_window))
        station_e_u = np.tile(ss_sys.station_e_u, reps=(1, n_interval_window))

        self.var_station_e_x = [array(self.variables.add(
            lb=station_e_l.ravel(), ub=station_e_u.ravel(),
            names=['e_station{0}_t{1}'.format(i_station, j_interval)
                   for i_station in range(n_station)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_station, n_interval_window)
            for s_scenario in range(n_scenario)]

        # model distribution system
        n_line = ds_sys.n_line
        n_bus = ds_sys.n_bus
        # Line active power, (n_line, n_interval_window)
        pij_l = np.tile(ds_sys.pij_l, reps=(1, n_interval_window))
        pij_u = np.tile(ds_sys.pij_u, reps=(1, n_interval_window))

        self.var_pij_x = [array(self.variables.add(
            lb=pij_l.ravel(), ub=pij_u.ravel(),
            names=['pij_l{0}_t{1}'.format(i_line, j_interval)
                   for i_line in range(n_line)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_line, n_interval_window)
            for s_scenario in range(n_scenario)]

        # Line reactive power,  (n_line, n_interval_window)
        qij_l = np.tile(ds_sys.qij_l, reps=(1, n_interval_window))
        qij_u = np.tile(ds_sys.qij_u, reps=(1, n_interval_window))

        self.var_qij_x = [array(self.variables.add(
            lb=qij_l.ravel(), ub=qij_u.ravel(),
            names=['qij_l{0}_t{1}'.format(i_line, j_interval)
                   for i_line in range(n_line)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_line, n_interval_window)
            for s_scenario in range(n_scenario)]

        # bus voltage, (n_bus, n_interval_window)
        vm_l = np.tile(ds_sys.vm_l, reps=(1, n_interval_window))
        vm_u = np.tile(ds_sys.vm_u, reps=(1, n_interval_window))

        self.var_vm_x = [array(self.variables.add(
            lb=vm_l.ravel(), ub=vm_u.ravel(),
            names=['vm_l{0}_t{1}'.format(i_bus, j_interval)
                   for i_bus in range(n_bus)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_bus, n_interval_window)
            for s_scenario in range(n_scenario)]

        # aggregated active power generation at DS level,
        # (n_station, n_interval_window)
        self.var_aggregate_pg_x = [array(self.variables.add(
            lb=-cpx.infinity * np.ones((n_station, n_interval_window)).ravel(),
            names=['aggregate_pg{0}_t{1}'.format(i_station, j_interval)
                   for i_station in range(n_station)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_station, n_interval_window)
            for s_scenario in range(n_scenario)]

        # aggregated reactive power generation at DS level,
        # (n_station, n_interval_window)
        # lb = full((n_station, n_interval_window), fill_value=-inf).ravel()
        self.var_aggregate_qg_x = [array(self.variables.add(
            lb=-cpx.infinity * np.ones((n_station, n_interval_window)).ravel(),
            names=['aggregate_qg{0}_t{1}'.format(i_station, j_interval)
                   for i_station in range(n_station)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_station, n_interval_window)
            for s_scenario in range(n_scenario)]

        # sign for load restoration, (n_load, n_interval_window)

        n_load = ds_sys.n_load
        # self.variables.type.binary is equivalent to 'B'
        # self.var_gama_load_x = array(self.variables.add(
        #     types=['B'] * (n_load * n_interval_window),
        #     names=['gama_load{0}_t{1}'.format(i_d, j_interval)
        #            for i_d in range(n_load)
        #            for j_interval in range(n_interval_window)])
        # ).reshape(n_load, n_interval_window)

        # active load restoration,
        # Set the load in the first interval in this time window to realization
        # !!! pload_scenario[s_scenario, :, interval] shape is changed!
        self.var_pd_x = [array(self.variables.add(
            # ub=pload_scenario[s_scenario, :, interval].ravel(),
            # ub1=ds_sys.pload[:, interval].ravel(),
            ub=pload_scenario[s_scenario, ...][:, interval].ravel(),
            names=['p_load{0}_t{1}'.format(i_d, j_interval)
                   for i_d in range(n_load)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_load, n_interval_window)
            for s_scenario in range(n_scenario)]

        # reactive load restoration,
        self.var_qd_x = [array(self.variables.add(
            ub=qload_scenario[s_scenario][:, interval].ravel(),
            names=['q_load{0}_t{1}'.format(i_d, j_interval)
                   for i_d in range(n_load)
                   for j_interval in range(n_interval_window)])
            ).reshape(n_load, n_interval_window)
            for s_scenario in range(n_scenario)]

        # Line connection status, 1-D array, (n_line)
        self.var_alpha_branch_x = [array(self.variables.add(
            types=['B'] * n_line,
            names=['alpha_branch{0}'.format(i_line)
                   for i_line in range(n_line)])
            ).reshape(n_line)
            for s_scenario in range(n_scenario)]

        # Auxiliary variables for line status, 1-D array, (n_line)
        self.var_betaij_x = [array(self.variables.add(
            types=['B'] * n_line,
            names=['betaij_{0}'.format(i_line)
                   for i_line in range(n_line)])
            ).reshape(n_line)
            for s_scenario in range(n_scenario)]

        self.var_betaji_x = [array(self.variables.add(
            types=['B'] * n_line,
            names=['betaji_{0}'.format(i_line)
                   for i_line in range(n_line)])
            ).reshape(n_line)
            for s_scenario in range(n_scenario)]

        pass
        # variables for scenario, 1-D array, (n_scenario)
        # var_scenario = array(model_x.variables.add(
        #     types=['B'] * n_scenario,
        #     names=['scenario_{0}'.format(i_scenario)
        #            for i_scenario in range(n_scenario)])
        # ).reshape(n_scenario)

        #  The total number of variables
        # self.variables.get_num()

    def add_objectives(self, time_sys, scenario_sys, ds_sys, ss_sys, tess_sys, tsn_sys):
        # Add Objective function
        # In comparison with a = zip(var_sign_onroad_x.tolist(),
        # [cost_ev_transit] * var_sign_onroad_x.size)
        # transportation cost (n_tess, n_interval_window)
        # zip(a, b), a is index and must conduct .tolist() but
        # b doesn't need to do .tolist()
        import numpy as np
        import copy

        n_interval_window = time_sys.n_interval_window
        interval = time_sys.interval
        delta_t = time_sys.delta_t

        n_scenario = scenario_sys.n_scenario
        scenario_weight = scenario_sys.scenario_weight

        pload_realization = copy.deepcopy(scenario_sys.pload_realization)
        qload_realization = copy.deepcopy(scenario_sys.qload_realization)
        pload_scenario = copy.deepcopy(scenario_sys.pload_scenario)
        qload_scenario = copy.deepcopy(scenario_sys.qload_scenario)

        pload_scenario[:, :, interval[0]] = np.tile(pload_realization[:, interval[0]], reps=(n_scenario, 1))
        qload_scenario[:, :, interval[0]] = np.tile(qload_realization[:, interval[0]], reps=(n_scenario, 1))

        tess_cost_transportation = np.tile(tess_sys.tess_cost_transportation,
                                           reps=(1, n_interval_window))
        for s_scenario in range(n_scenario):
            self.objective.set_linear(zip(self.var_sign_onroad_x[s_scenario].ravel().tolist(),
                            (tess_cost_transportation * scenario_weight[s_scenario]).ravel()))

        # charging cost, (n_tess, n_interval_window)
        sn_mva = ds_sys.sn_mva
        # note the dimension mismatch between var_tess2st_pch_x and
        # tess_cost_power, so make corrections !!!
        tess_cost_power = np.tile(tess_sys.tess_cost_power[:, np.newaxis, :],
                                  reps=(1, ss_sys.n_station, 1))
        for s_scenario in range(n_scenario):
            self.objective.set_linear(zip(self.var_tess2st_pch_x[s_scenario].ravel().tolist(),
                                      (tess_cost_power * delta_t * sn_mva * scenario_weight[s_scenario]).ravel()))

        # # discharging cost
            self.objective.set_linear(zip(self.var_tess2st_pdch_x[s_scenario].ravel().tolist(),
                                      (tess_cost_power * delta_t * sn_mva * scenario_weight[s_scenario]).ravel()))

        # generation cost
        station_gencost = np.tile(ss_sys.station_gencost,
                                  reps=(1, n_interval_window))

        for s_scenario in range(n_scenario):
            self.objective.set_linear(zip(self.var_station_p_x[s_scenario].ravel().tolist(),
                                      (station_gencost * delta_t * sn_mva * scenario_weight[s_scenario]).ravel()))

        # load interruption cost
        load_interruption_cost = np.tile(ds_sys.load_interruption_cost,
                                         reps=(1, n_interval_window))
        for s_scenario in range(n_scenario):
            self.objective.set_linear(
                zip(self.var_pd_x[s_scenario].ravel().tolist(),
                    (-load_interruption_cost * delta_t * sn_mva * scenario_weight[s_scenario]).ravel()))

        # Add objective constant
        objective_offset = 0
        for s_scenario in range(n_scenario):
            objective_offset += (
                (load_interruption_cost * pload_scenario[s_scenario][:, interval] *
                 delta_t * sn_mva * scenario_weight[s_scenario]).ravel().sum())

        self.objective.set_offset(objective_offset)
        # self.objective.set_offset(1)

        # Set objective sense
        self.objective.set_sense(self.objective.sense.minimize)

    def add_constraints(self, time_sys, scenario_sys, ds_sys, ss_sys, tess_sys, tsn_sys, result_list):
        # Form matrix A, vector b and sense
        # It turns to be more efficient to set up A incrementally as coo_matrix
        # while b as list
        # aeq is ili_matrix while beq is nd-array
        # time_start = time.time()  # zeros- 7s, roughly, lil_matrix- 0.7s, roughly
        import scipy.sparse as spar
        import copy
        from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
        # Extract parameters
        n_all_vars_x = self.variables.get_num()
        n_interval_window = time_sys.n_interval_window
        interval = time_sys.interval

        n_scenario = scenario_sys.n_scenario

        pload_realization = copy.deepcopy(scenario_sys.pload_realization)
        qload_realization = copy.deepcopy(scenario_sys.qload_realization)
        pload_scenario = copy.deepcopy(scenario_sys.pload_scenario)
        qload_scenario = copy.deepcopy(scenario_sys.qload_scenario)

        #
        pload_scenario[:, :, interval[0]] = np.tile(
            pload_realization[:, interval[0]], reps=(n_scenario, 1))
        qload_scenario[:, :, interval[0]] = np.tile(
            qload_realization[:, interval[0]], reps=(n_scenario, 1))


        n_tess = tess_sys.n_tess
        # Initialization of matrix A
        model_x_matrix_a = coo_matrix((0, n_all_vars_x))
        model_x_matrix_a = coo_matrix((0, n_all_vars_x))
        model_x_rhs = []
        model_x_senses = []

        # --------------------------------------------------------------------------
        # Retrieve the active arcs from the first interval of the last rolling loop
        n_tsn_arc = tsn_sys.n_tsn_arc
        tsn_cut_set = tsn_sys.tsn_cut_set

        # # Judge if there is previous result
        # if ds_sys.interval[0]:
        #     # If so, copy previous results to this window
        #     j_interval_previous = ds_sys.interval_window[0] - 1
        #
        #     aeq_previousarc = lil_matrix((n_tess, n_all_vars_x))
        #     beq_previousarc = np.ones((n_tess))
        #
        #     for i_tess in range(n_tess):
        #         # to find previous interval's active arc
        #         idx_active_arc = np.flatnonzero(
        #             np.logical_and(
        #                 result_list[j_interval_previous].res_tess_arc_x[i_tess],
        #                 tsn_cut_set[i_tess][:, j_interval_previous]))
        #
        #         aeq_previousarc[i_tess,
        #                         self.var_tess_arc_x[i_tess][idx_active_arc]] = 1
        #
        #     model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_previousarc])
        #     model_x_rhs += beq_previousarc.tolist()
        #     model_x_senses += ['E'] * n_tess

        # --------------------------------------------------------------------------
        # Each tess only in one status in each interval
        aeq_onestatus = lil_matrix((n_scenario * n_tess * n_interval_window, n_all_vars_x))
        beq_onestatus = np.ones((n_scenario * n_tess * n_interval_window))

        # retrieve parameters
        # tsn_cut_set_window = [tsn_cut_set[i_tess][:, ds_sys.interval_window]
        #                       for i_tess in range(n_tess)]

        for s_scenario in range(n_scenario):
            for i_tess in range(n_tess):
                for j_interval_window in range(n_interval_window):
                    aeq_onestatus[s_scenario*n_tess*n_interval_window +
                                  i_tess*n_interval_window + j_interval_window,
                        self.var_tess_arc_x[s_scenario][i_tess][tsn_cut_set[i_tess][:, j_interval_window]]] = 1

        # model_x_matrix_a = concatenate((model_x_matrix_a, aeq_onestatus), axis=0)
        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_onestatus])
        # # model_x_rhs = hstack([model_x_rhs, beq_onestatus])
        model_x_rhs += beq_onestatus.tolist()
        model_x_senses += ['E'] * n_scenario * n_tess * n_interval_window
        ''''''
        # --------------------------------------------------------------------------
        # Constraints for tess transit flow
        # Retrieve the tsn node in ascending order
        timepoint_window = time_sys.timepoint_window
        tsn_node = tsn_sys.tsn_node
        n_tsn_node = tsn_sys.n_tsn_node

        n_tsn_node_all = np.array(n_tsn_node).sum()

        # retrieve parameters
        NODE_I = tsn_sys.NODE_I
        NODE_T = tsn_sys.NODE_T
        NODE_L = tsn_sys.NODE_L

        tsn_f2arc = tsn_sys.tsn_f2arc
        tsn_t2arc = tsn_sys.tsn_t2arc

        aeq_transitflow = lil_matrix((n_scenario * n_tsn_node_all, n_all_vars_x))
        beq_transitflow = np.zeros(n_scenario * n_tsn_node_all)

        for s_scenario in range(n_scenario):
            for i_tess in range(n_tess):
                # node flow balance only holds for tsn nodes from time point
                # (in window) 1 to last-to-second
                tsn_node_tess_balance = tsn_node[i_tess][
                    (tsn_node[i_tess][:, NODE_T] >= timepoint_window[1]) & \
                    (tsn_node[i_tess][:, NODE_T] < timepoint_window[-1]),
                    NODE_I
                ]

                aeq_transitflow[np.ix_(s_scenario * n_tsn_node_all +
                    np.array(n_tsn_node[:i_tess]).sum() +
                    tsn_node_tess_balance, self.var_tess_arc_x[s_scenario][i_tess])] = \
                    tsn_t2arc[i_tess][tsn_node_tess_balance, :] - \
                    tsn_f2arc[i_tess][tsn_node_tess_balance, :]
                # !!!! error has been corrected
                # aeq_transitflow[ix_(i_tess*n_tsn_node+j_tsn_node_range,
                # var_ev_arc_x[i_tess, :])] =

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_transitflow])
        model_x_rhs += beq_transitflow.tolist()
        model_x_senses += ['E'] * n_scenario * n_tsn_node_all

        # --------------------------------------------------------------------------
        # charging/discharging with respect to position
        # (n_tess) list of (n_tsn_arc[i], n_interval_window)
        S_NODE_L = tsn_sys.S_NODE_L
        T_NODE_L = tsn_sys.T_NODE_L

        tsn_holding_arc = tsn_sys.tsn_holding_arc

        n_station = ss_sys.n_station
        idx_microgrid = ss_sys.idx_microgrid
        n_microgrid = ss_sys.n_microgrid


        aineq_pchposition = lil_matrix(
            (n_scenario * n_tess * n_interval_window * n_station, n_all_vars_x))
        bineq_pchposition = np.zeros(n_scenario * n_tess * n_interval_window * n_station)
        aineq_pdchposition = lil_matrix(
            (n_scenario * n_tess * n_interval_window * n_station, n_all_vars_x))
        bineq_pdchposition = np.zeros(n_scenario * n_tess * n_interval_window * n_station)

        for s_scenario in range(n_scenario):
            for i_tess in range(n_tess):
                for j_interval_window in range(n_interval_window):
                    # pch <= ***Pch,max
                    aineq_pchposition[np.ix_(s_scenario*n_tess*n_interval_window*n_station +
                        i_tess*n_interval_window*n_station + j_interval_window*n_station
                        + np.arange(n_station),
                        self.var_tess2st_pch_x[s_scenario][i_tess, :, j_interval_window])] = \
                        np.eye(n_station)
                    # !!! need to correct... It's ok now.
                    # Check if there are holding arcs for the interval
                    # If so, enter it, otherwise skip it
                    if tsn_holding_arc[i_tess][:, j_interval_window].any():
                        # normally there are four holding arcs in a interval

                        # in case the initial interval in the window, only one holding arcs
                        these_tsn_arc = tsn_sys.tsn_arc[i_tess][
                            tsn_holding_arc[i_tess][:, j_interval_window]]

                        assert (these_tsn_arc[:, S_NODE_L] == these_tsn_arc[:, T_NODE_L]).all()
                        # the location in station system number
                        these_node_location = these_tsn_arc[:, S_NODE_L]
                        n_these_node = these_node_location.size

                        aineq_pchposition[np.ix_(s_scenario*n_tess*n_interval_window*n_station +
                            i_tess * n_interval_window * n_station
                                   + j_interval_window * n_station + these_node_location,
                                   self.var_tess_arc_x[s_scenario][i_tess][
                                       tsn_holding_arc[i_tess][:,
                                       j_interval_window]])] \
                            = -tess_sys.tess_pch_u[i_tess] * np.eye(n_these_node)

                        # try:
                        #     aineq_pchposition[np.ix_(i_tess * n_interval_window * n_station
                        #     + j_interval_window * n_station + idx_microgrid,
                        #     self.var_tess_arc_x[i_tess][tsn_holding_arc[i_tess][:,j_interval_window]])] \
                        #     = -tess_sys.tess_pch_u[i_tess] * np.eye(n_microgrid)
                        # except:
                        #     pass

                    # Pdch <= ***Pdch,max
                    aineq_pdchposition[np.ix_(s_scenario*n_tess*n_interval_window*n_station +
                        i_tess * n_interval_window * n_station
                        + j_interval_window * n_station + np.arange(n_station),
                        self.var_tess2st_pdch_x[s_scenario][i_tess, :, j_interval_window])] = \
                        np.eye(n_station)

                    if tsn_holding_arc[i_tess][:, j_interval_window].any():
                        these_tsn_arc = tsn_sys.tsn_arc[i_tess][tsn_holding_arc[i_tess][:, j_interval_window]]

                        assert (these_tsn_arc[:, S_NODE_L] == these_tsn_arc[:, T_NODE_L]).all()
                        # the location in station system number
                        these_node_location = these_tsn_arc[:, S_NODE_L]
                        n_these_node = these_node_location.size

                        aineq_pdchposition[np.ix_(s_scenario*n_tess*n_interval_window*n_station +
                            i_tess * n_interval_window * n_station
                            + j_interval_window * n_station + these_node_location,
                            self.var_tess_arc_x[s_scenario][i_tess][tsn_holding_arc[i_tess][:, j_interval_window]])] \
                        = -tess_sys.tess_pdch_u[i_tess] * np.eye(n_these_node)

                        # aineq_pdchposition[np.ix_(i_tess * n_interval_window * n_station
                        #     + j_interval_window * n_station + idx_microgrid,
                        #     self.var_tess_arc_x[i_tess][tsn_holding_arc[i_tess][:, j_interval_window]])] \
                        # = -tess_sys.tess_pdch_u[i_tess] * np.eye(n_microgrid)

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aineq_pchposition,
                                        aineq_pdchposition])
        model_x_rhs += (
                    bineq_pchposition.tolist() + bineq_pdchposition.tolist())
        model_x_senses += ['L'] * 2 * n_scenario * n_tess * n_interval_window * n_station

        # --------------------------------------------------------------------------
        # charging/discharging with respect to battery status
        aineq_pchstatus = lil_matrix((n_scenario * n_tess * n_interval_window, n_all_vars_x))
        bineq_pchstatus = np.zeros(n_scenario * n_tess * n_interval_window)

        aineq_pdchstatus = lil_matrix(
            (n_scenario * n_tess * n_interval_window, n_all_vars_x))
        bineq_pdchstatus = np.zeros(n_scenario * n_tess * n_interval_window)

        for s_scenario in range(n_scenario):
            for i_tess in range(n_tess):
                for j_interval_window in range(n_interval_window):
                    aineq_pchstatus[s_scenario*n_tess*n_interval_window +
                        i_tess*n_interval_window + j_interval_window,
                        self.var_tess2st_pch_x[s_scenario][i_tess, :, j_interval_window]] = 1

                    aineq_pchstatus[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                        self.var_sign_ch_x[s_scenario][i_tess, j_interval_window]] = \
                        -tess_sys.tess_pch_u[i_tess]

                    aineq_pdchstatus[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                        self.var_tess2st_pdch_x[s_scenario][i_tess, :, j_interval_window]] = 1

                    aineq_pdchstatus[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                        self.var_sign_dch_x[s_scenario][i_tess, j_interval_window]] = \
                        -tess_sys.tess_pdch_u[i_tess]

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aineq_pchstatus,
                                        aineq_pdchstatus])
        model_x_rhs += (bineq_pchstatus.tolist() + bineq_pdchstatus.tolist())
        model_x_senses += ['L'] * 2 * n_scenario * n_tess * n_interval_window

        # --------------------------------------------------------------------------
        # charging/discharging status
        aineq_chdchstatus = lil_matrix(
            (n_scenario * n_tess * n_interval_window, n_all_vars_x))
        bineq_chdchstatus = np.zeros(n_scenario * n_tess * n_interval_window)

        for s_scenario in range(n_scenario):
            for i_tess in range(n_tess):
                for j_interval_window in range(n_interval_window):
                    aineq_chdchstatus[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                            self.var_sign_ch_x[s_scenario][i_tess, j_interval_window]] = 1

                    aineq_chdchstatus[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                            self.var_sign_dch_x[s_scenario][i_tess, j_interval_window]] = 1

                    # check if there are holding arcs in the interval
                    if tsn_holding_arc[i_tess][:, j_interval_window].any():
                        aineq_chdchstatus[s_scenario*n_tess*n_interval_window +
                            i_tess * n_interval_window + j_interval_window,
                            self.var_tess_arc_x[s_scenario][i_tess][tsn_holding_arc[i_tess][:,
                            j_interval_window]]] = -1

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aineq_chdchstatus])
        model_x_rhs += bineq_chdchstatus.tolist()
        model_x_senses += ['L'] * n_scenario * n_tess * n_interval_window

        # --------------------------------------------------------------------------
        # constraint for sign_onroad
        aeq_signonroad = lil_matrix((n_scenario * n_tess * n_interval_window, n_all_vars_x))
        beq_signonroad = np.ones(n_scenario * n_tess * n_interval_window)

        for s_scenario in range(n_scenario):
            for i_tess in range(n_tess):
                for j_interval_window in range(n_interval_window):
                    aeq_signonroad[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                        self.var_sign_onroad_x[s_scenario][i_tess, j_interval_window]] = 1
                    # check if there are holding arcs in the interval
                    if tsn_holding_arc[i_tess][:, j_interval_window].any():
                        aeq_signonroad[s_scenario*n_tess*n_interval_window +
                            i_tess * n_interval_window + j_interval_window,
                            self.var_tess_arc_x[s_scenario][i_tess][tsn_holding_arc[i_tess][:,
                                                        j_interval_window]]] = 1

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_signonroad])
        model_x_rhs += beq_signonroad.tolist()
        model_x_senses += ['E'] * n_scenario * n_tess * n_interval_window

        # --------------------------------------------------------------------------
        # constraints for energy of tess
        delta_t = time_sys.delta_t
        interval = time_sys.interval

        aeq_energy = lil_matrix((n_scenario * n_tess * n_interval_window, n_all_vars_x))
        beq_energy = np.zeros(n_scenario * n_tess * n_interval_window)
        for s_scenario in range(n_scenario):
            for i_tess in range(n_tess):
                for j_interval_window in range(n_interval_window - 1):
                    aeq_energy[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                            self.var_tess_e_x[s_scenario][i_tess, j_interval_window + 1]] = 1
                    aeq_energy[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                            self.var_tess_e_x[s_scenario][i_tess, j_interval_window]] = -1

                    aeq_energy[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                               self.var_tess2st_pch_x[s_scenario][i_tess, :, j_interval_window + 1]] \
                        = -delta_t * tess_sys.tess_ch_efficiency[i_tess]
                    aeq_energy[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + j_interval_window,
                               self.var_tess2st_pdch_x[s_scenario][i_tess, :, j_interval_window + 1]] \
                        = delta_t / tess_sys.tess_dch_efficiency[i_tess]

                # Considering intial status, it is stored at the end
                # !!!

                aeq_energy[s_scenario*n_tess*n_interval_window +
                    i_tess * n_interval_window + n_interval_window - 1,
                           self.var_tess_e_x[s_scenario][i_tess, 0]] = 1
                aeq_energy[s_scenario*n_tess*n_interval_window +
                    i_tess * n_interval_window + n_interval_window - 1,
                           self.var_tess2st_pch_x[s_scenario][i_tess, :, 0]] \
                    = -delta_t * tess_sys.tess_ch_efficiency[i_tess]
                aeq_energy[s_scenario*n_tess*n_interval_window +
                    i_tess * n_interval_window + n_interval_window - 1,
                           self.var_tess2st_pdch_x[s_scenario][i_tess, :, 0]] \
                    = delta_t / tess_sys.tess_dch_efficiency[i_tess]

                if interval[0]:
                    # if the initial interval in the window is not 0, it needs to
                    # retrieve from last model.
                    j_interval_previous = interval[0] - 1
                    beq_energy[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + n_interval_window - 1] \
                        = result_list[j_interval_previous].realres_tess_e_x[i_tess]
                else:
                    # if the initial interval in the window is 0,
                    beq_energy[s_scenario*n_tess*n_interval_window +
                        i_tess * n_interval_window + n_interval_window - 1] \
                        = tess_sys.tess_e_init[i_tess]

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_energy])
        model_x_rhs += beq_energy.tolist()
        model_x_senses += ['E'] * n_scenario * n_tess * n_interval_window

        # --------------------------------------------------------------------------
        # constraints for generation of station
        aeq_generationp = lil_matrix(
            (n_scenario * n_station * n_interval_window, n_all_vars_x))
        beq_generationp = np.zeros(n_scenario * n_station * n_interval_window)

        aeq_generationq = lil_matrix(
            (n_scenario * n_station * n_interval_window, n_all_vars_x))
        beq_generationq = np.zeros(n_scenario * n_station * n_interval_window)

        for s_scenario in range(n_scenario):
            for i_station in range(n_station):
                for j_interval_window in range(n_interval_window):
                    # pch
                    aeq_generationp[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_tess2st_pch_x[s_scenario][:, i_station, j_interval_window]] = 1
                    # pdch
                    aeq_generationp[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_tess2st_pdch_x[s_scenario][:, i_station, j_interval_window]] = -1
                    # station_p
                    aeq_generationp[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_station_p_x[s_scenario][i_station, j_interval_window]] = -1
                    # aggregated_pg
                    aeq_generationp[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_aggregate_pg_x[s_scenario][i_station, j_interval_window]] = 1

                    # station_q
                    aeq_generationq[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_station_q_x[s_scenario][i_station, j_interval_window]] = -1
                    # aggregated_q
                    aeq_generationq[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_aggregate_qg_x[s_scenario][i_station, j_interval_window]] = 1

        model_x_matrix_a = spar.vstack(
            [model_x_matrix_a, aeq_generationp, aeq_generationq])
        model_x_rhs += beq_generationp.tolist() + beq_generationq.tolist()
        model_x_senses += ['E'] * 2 * n_scenario * n_station * n_interval_window

        # --------------------------------------------------------------------------
        # The amount of energy for each station in the time point t (end point of
        # time interval t)
        aeq_mgenergy = lil_matrix((n_scenario * n_station * n_interval_window, n_all_vars_x))
        beq_mgenergy = np.zeros(n_scenario * n_station * n_interval_window)

        for s_scenario in range(n_scenario):
            for i_station in range(n_station):
                for j_interval_window in range(n_interval_window - 1):
                    # Estation_t+1
                    aeq_mgenergy[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_station_e_x[s_scenario][i_station, j_interval_window + 1]] = 1

                    # Estation_t
                    aeq_mgenergy[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_station_e_x[s_scenario][i_station, j_interval_window]] = -1

                    # station_p
                    aeq_mgenergy[s_scenario*n_station*n_interval_window +
                        i_station * n_interval_window + j_interval_window,
                        self.var_station_p_x[s_scenario][i_station, j_interval_window + 1]] = delta_t
                # For the initial interval
                aeq_mgenergy[s_scenario*n_station*n_interval_window +
                    (i_station + 1) * n_interval_window - 1,
                             self.var_station_e_x[s_scenario][i_station, 0]] = 1
                aeq_mgenergy[s_scenario*n_station*n_interval_window +
                    (i_station + 1) * n_interval_window - 1,
                             self.var_station_p_x[s_scenario][i_station, 0]] = delta_t

                if interval[0]:
                    # if the interval in entire horizon is not the first interval
                    j_interval_previous = interval[0] - 1
                    beq_mgenergy[s_scenario*n_station*n_interval_window +
                        (i_station + 1) * n_interval_window - 1] \
                        = result_list[j_interval_previous].realres_station_e_x[
                        i_station]
                else:
                    # if the initial interval in the entire horizon
                    beq_mgenergy[s_scenario*n_station*n_interval_window +
                        (i_station + 1) * n_interval_window - 1] \
                        = ss_sys.station_e_u[i_station, 0]

        model_x_matrix_a = spar.vstack(
            [model_x_matrix_a, aeq_mgenergy])  # error because of indentation
        model_x_rhs += beq_mgenergy.tolist()
        model_x_senses += ['E'] * n_scenario * n_station * n_interval_window

        # --------------------------------------------------------------------------
        # topology constraint, |N|-|M|
        n_bus = ds_sys.n_bus
        n_microgrid = ss_sys.n_microgrid
        aeq_dstree = lil_matrix((n_scenario, n_all_vars_x))
        beq_dstree = np.zeros(n_scenario)

        for s_scenario in range(n_scenario):
            aeq_dstree[s_scenario, self.var_alpha_branch_x[s_scenario]] = 1
            beq_dstree[s_scenario] = n_bus - n_microgrid  # todo ?? if bus deleted?

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dstree])
        model_x_rhs += beq_dstree.tolist()
        model_x_senses += ['E'] * n_scenario

        # --------------------------------------------------------------------------
        # topology constraint, bij + bji = alphaij
        n_line = ds_sys.n_line
        aeq_dsbranchstatus = lil_matrix((n_scenario * n_line, n_all_vars_x))
        beq_dsbranchstatus = np.zeros(n_scenario * n_line)

        for s_scenario in range(n_scenario):
            aeq_dsbranchstatus[np.ix_(s_scenario*n_line + np.arange(n_line),
                self.var_alpha_branch_x[s_scenario])] = -np.eye(n_line)
            aeq_dsbranchstatus[np.ix_(s_scenario*n_line + np.arange(n_line),
                self.var_betaij_x[s_scenario])] = np.eye(n_line)
            aeq_dsbranchstatus[np.ix_(s_scenario*n_line + np.arange(n_line),
                self.var_betaji_x[s_scenario])] = np.eye(n_line)

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dsbranchstatus])
        model_x_rhs += beq_dsbranchstatus.tolist()
        model_x_senses += ['E'] * n_scenario * n_line

        # --------------------------------------------------------------------------
        # topology constraint, exact one parent for each bus other than mg bus
        aeq_dsoneparent = lil_matrix((n_scenario * (n_bus - n_microgrid), n_all_vars_x))
        beq_dsoneparent = np.ones(n_scenario * (n_bus - n_microgrid))

        for s_scenario in range(n_scenario):
            aeq_dsoneparent[np.ix_(s_scenario*(n_bus - n_microgrid) + np.arange(n_bus - n_microgrid),
                self.var_betaij_x[s_scenario])] = ds_sys.incidence_ds_fbus2line[ds_sys.idx_pq_bus, :]
            aeq_dsoneparent[np.ix_(s_scenario*(n_bus - n_microgrid) + np.arange(n_bus - n_microgrid),
                self.var_betaji_x[s_scenario])] = ds_sys.incidence_ds_tbus2line[ds_sys.idx_pq_bus, :]

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dsoneparent])
        model_x_rhs += beq_dsoneparent.tolist()
        model_x_senses += ['E'] * n_scenario * (n_bus - n_microgrid)

        # --------------------------------------------------------------------------
        # topology constraint, mg buses has no parent
        aeq_dsnoparent = lil_matrix((n_scenario * n_microgrid, n_all_vars_x))
        beq_dsnoparent = np.zeros(n_scenario * n_microgrid)

        for s_scenario in range(n_scenario):
            aeq_dsnoparent[np.ix_(s_scenario*n_microgrid + np.arange(n_microgrid),
                self.var_betaij_x[s_scenario])] = ds_sys.incidence_ds_fbus2line[ds_sys.idx_ref_bus, :]
            aeq_dsnoparent[np.ix_(s_scenario*n_microgrid + np.arange(n_microgrid),
                self.var_betaji_x[s_scenario])] = ds_sys.incidence_ds_tbus2line[ds_sys.idx_ref_bus, :]

        # n_index_betaij = ds_sys.idx_beta_ij.shape[0]
        # n_index_betaji = ds_sys.idx_beta_ji.shape[0]
        # aeq_dsnoparent = lil_matrix(
        #     (n_index_betaij + n_index_betaji, n_all_vars_x))
        # beq_dsnoparent = np.zeros(n_index_betaij + n_index_betaji)
        #
        # # index_beta_ij is different of array and csr_matrix
        # aeq_dsnoparent[:n_index_betaij, self.var_betaij_x[ds_sys.idx_beta_ij]
        # ] = np.eye(n_index_betaij)
        # aeq_dsnoparent[n_index_betaij:, self.var_betaji_x[ds_sys.idx_beta_ji]
        # ] = np.eye(n_index_betaji)

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dsnoparent])
        model_x_rhs += beq_dsnoparent.tolist()
        model_x_senses += ['E'] * n_scenario * n_microgrid

        # -----------------------------------------------------------------------
        # power balance !!!
        aeq_dskclp = lil_matrix((n_scenario * n_interval_window * n_bus, n_all_vars_x))
        beq_dskclp = np.zeros(n_scenario * n_interval_window * n_bus)
        aeq_dskclq = lil_matrix((n_scenario * n_interval_window * n_bus, n_all_vars_x))
        beq_dskclq = np.zeros(n_scenario * n_interval_window * n_bus)

        for s_scenario in range(n_scenario):
            for j_interval_window in range(n_interval_window):
                # pij
                aeq_dskclp[np.ix_(s_scenario*n_interval_window*n_bus +
                    j_interval_window*n_bus + np.arange(n_bus),
                        self.var_pij_x[s_scenario][:, j_interval_window])] = (
                        ds_sys.incidence_ds_tbus2line -
                        ds_sys.incidence_ds_fbus2line)
                # aggregate generation p
                aeq_dskclp[np.ix_(s_scenario*n_interval_window*n_bus +
                    j_interval_window*n_bus + np.arange(n_bus),
                            self.var_aggregate_pg_x[s_scenario][:, j_interval_window])] \
                    = ss_sys.station2dsbus.T
                # p_load
                n_load = ds_sys.n_load
                aeq_dskclp[np.ix_(s_scenario*n_interval_window*n_bus +
                    j_interval_window * n_bus + np.arange(n_bus),
                                  self.var_pd_x[s_scenario][:, j_interval_window])] \
                    = -ds_sys.mapping_load2dsbus.T * np.eye(n_load)
                # qij
                aeq_dskclq[np.ix_(s_scenario*n_interval_window*n_bus +
                    j_interval_window * n_bus + np.arange(n_bus),
                                  self.var_qij_x[s_scenario][:, j_interval_window])] = (
                        ds_sys.incidence_ds_tbus2line -
                        ds_sys.incidence_ds_fbus2line)
                # aggregate generation q !!!!!!
                aeq_dskclq[np.ix_(s_scenario*n_interval_window*n_bus +
                    j_interval_window * n_bus + np.arange(n_bus),
                                  self.var_aggregate_qg_x[s_scenario][:, j_interval_window])] \
                    = ss_sys.station2dsbus.T
                # q_load
                aeq_dskclq[np.ix_(s_scenario*n_interval_window*n_bus +
                    j_interval_window * n_bus + np.arange(n_bus),
                                  self.var_qd_x[s_scenario][:, j_interval_window])] \
                    = -ds_sys.mapping_load2dsbus.T * np.eye(n_load)

        model_x_matrix_a = spar.vstack(
            [model_x_matrix_a, aeq_dskclp, aeq_dskclq])
        model_x_rhs += beq_dskclp.tolist() + beq_dskclq.tolist()
        model_x_senses += ['E'] * 2 * n_scenario * n_bus * n_interval_window

        # --------------------------------------------------------------------------
        # KVL with branch status
        large_m = 1e6
        n_line = ds_sys.n_line
        v0 = ds_sys.v0
        branch_r = ds_sys.ppnet.line['branch_r_pu']
        branch_x = ds_sys.ppnet.line['branch_x_pu']

        aineq_dskvl_u = lil_matrix((n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dskvl_u = large_m * np.ones(n_scenario * n_line * n_interval_window)
        aineq_dskvl_l = lil_matrix((n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dskvl_l = large_m * np.ones(n_scenario * n_line * n_interval_window)

        for s_scenario in range(n_scenario):
            for j_interval_window in range(n_interval_window):
                # v_j^t - v_i^t <= M(1-alphabranch) + (rij*pij + xij*qij) / v0
                aineq_dskvl_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = -np.diag(branch_r) / v0

                aineq_dskvl_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = -np.diag(branch_x) / v0

                aineq_dskvl_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_vm_x[s_scenario][:, j_interval_window])] = (
                        ds_sys.incidence_ds_tbus2line.T -
                        ds_sys.incidence_ds_fbus2line.T).toarray()

                aineq_dskvl_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = large_m * np.eye(n_line)

                # v_j^t - v_i^t >= -M(1-alphabranch) + (rij*pij + xij*qij) / v0
                aineq_dskvl_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = np.diag(branch_r) / v0

                aineq_dskvl_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = np.diag(branch_x) / v0

                aineq_dskvl_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_vm_x[s_scenario][:, j_interval_window])] = (
                        ds_sys.incidence_ds_fbus2line.T -
                        ds_sys.incidence_ds_tbus2line.T).toarray()

                aineq_dskvl_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = large_m * np.eye(n_line)

        model_x_matrix_a = spar.vstack(
            [model_x_matrix_a, aineq_dskvl_u, aineq_dskvl_l])
        model_x_rhs += bineq_dskvl_u.tolist() + bineq_dskvl_l.tolist()
        model_x_senses += ['L'] * 2 * n_scenario * n_line * n_interval_window

        # # set v_j^t - v_i^t == -M(1-alphabranch) + (rij*pij + xij*qij) / v0 for
        # # indicator constraints
        # # This part is added separately after the entire model
        # n_line = ds_sys.n_line
        # v0 = ds_sys.v0
        # branch_r = ds_sys.ppnet.line['branch_r_pu']
        # branch_x = ds_sys.ppnet.line['branch_x_pu']
        #
        # aeq_dskvl_e = lil_matrix((n_scenario * n_line * n_interval_window, n_all_vars_x))
        # beq_dskvl_e = np.zeros(n_scenario * n_line * n_interval_window)
        # indicator_dskvl_e = lil_matrix((n_scenario * n_line * n_interval_window, 1))
        # complemented_dskvl_e = [1] * n_line * n_interval_window
        #
        # for j_interval_window in range(n_interval_window):
        #     # v_j^t - v_i^t <= M(1-alphabranch) + (rij*pij + xij*qij) / v0
        #     aeq_dskvl_e[np.ix_(j_interval_window * n_line + np.arange(n_line),
        #         self.var_pij_x[:, j_interval_window])] = -np.diag(branch_r) / v0
        #
        #     aeq_dskvl_e[np.ix_(j_interval_window * n_line + np.arange(n_line),
        #         self.var_qij_x[:, j_interval_window])] = -np.diag(branch_x) / v0
        #
        #     aeq_dskvl_e[np.ix_(j_interval_window * n_line + np.arange(n_line),
        #         self.var_vm_x[:, j_interval_window])] = (
        #             ds_sys.incidence_ds_tbus2line.T -
        #             ds_sys.incidence_ds_fbus2line.T).toarray()
        #
        #     # aineq_dskvl_u[np.ix_(j_interval_window * n_line + np.arange(n_line),
        #     #                      self.var_alpha_branch_x)] = large_m * np.eye(
        #     #     n_line)
        #     indicator_dskvl_e[j_interval_window * n_line + np.arange(n_line), 0] \
        #         = self.var_alpha_branch_x[:, np.newaxis]
        #
        # aeq_dskvl_e = aeq_dskvl_e.toarray().tolist()
        # beq_dskvl_e = beq_dskvl_e.tolist()
        # indicator_dskvl_e = indicator_dskvl_e.toarray().ravel().tolist()
        # sense_dskvl_e = ['E'] * n_line * n_interval_window
        # # indtype_dskvl_e =

        # branch power limit pij and qij, respectively
        slmax = ds_sys.slmax
        # alphabranch * -slmax <= pij <= alphabranch * slmax
        aineq_dspij_u = lil_matrix((n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dspij_u = np.zeros(n_scenario * n_line * n_interval_window)
        aineq_dspij_l = lil_matrix((n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dspij_l = np.zeros(n_scenario * n_line * n_interval_window)
        # alphabranch * -slmax <= qij <= alphabranch * slmax
        aineq_dsqij_u = lil_matrix((n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dsqij_u = np.zeros(n_scenario * n_line * n_interval_window)
        aineq_dsqij_l = lil_matrix((n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dsqij_l = np.zeros(n_scenario * n_line * n_interval_window)

        for s_scenario in range(n_scenario):
            for j_interval_window in range(n_interval_window):
                # pij - alphabranch * slmax <= 0
                aineq_dspij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = np.eye(n_line)

                aineq_dspij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = -np.diag(slmax)
                # -pij - alphabranch * slmax <= 0
                aineq_dspij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = -np.eye(n_line)

                aineq_dspij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = -np.diag(slmax)
                # qij - alphabranch * slmax <= 0
                aineq_dsqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = np.eye(n_line)

                aineq_dsqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = -np.diag(slmax)

                # -qij - alphabranch * slmax <= 0
                aineq_dsqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = -np.eye(n_line)

                aineq_dsqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = -np.diag(slmax)

        model_x_matrix_a = spar.vstack(
            [model_x_matrix_a, aineq_dspij_u, aineq_dspij_l, aineq_dsqij_u,
             aineq_dsqij_l])
        model_x_rhs += bineq_dspij_u.tolist() + bineq_dspij_l.tolist() + \
                       bineq_dsqij_u.tolist() + bineq_dsqij_l.tolist()
        model_x_senses += ['L'] * 4 * n_scenario * n_line * n_interval_window

        # Branch power limit pij + qij and pij - qij
        # *** <= pij + qij <= ***
        aineq_dspijaddqij_u = lil_matrix(
            (n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dspijaddqij_u = np.zeros(n_scenario * n_line * n_interval_window)
        aineq_dspijaddqij_l = lil_matrix(
            (n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dspijaddqij_l = np.zeros(n_scenario * n_line * n_interval_window)
        # *** <= pij - qij <= ***
        aineq_dspijsubqij_u = lil_matrix(
            (n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dspijsubqij_u = np.zeros(n_scenario * n_line * n_interval_window)
        aineq_dspijsubqij_l = lil_matrix(
            (n_scenario * n_line * n_interval_window, n_all_vars_x))
        bineq_dspijsubqij_l = np.zeros(n_scenario * n_line * n_interval_window)

        for s_scenario in range(n_scenario):
            for j_interval_window in range(n_interval_window):
                # pij + qij <= ***
                aineq_dspijaddqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = np.eye(n_line)

                aineq_dspijaddqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = np.eye(n_line)

                aineq_dspijaddqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line +
                    np.arange(n_line), self.var_alpha_branch_x[s_scenario])] = -np.sqrt(2) * np.diag(slmax)

                # *** < pij + qij
                aineq_dspijaddqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = -np.eye(n_line)

                aineq_dspijaddqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = -np.eye(n_line)

                aineq_dspijaddqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = -np.sqrt(2) * np.diag(slmax)

                # pij - qij <= ***
                aineq_dspijsubqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = np.eye(n_line)

                aineq_dspijsubqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = -np.eye(n_line)

                aineq_dspijsubqij_u[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = -np.sqrt(2) * np.diag(slmax)

                # *** < pij - qij
                aineq_dspijsubqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_pij_x[s_scenario][:, j_interval_window])] = -np.eye(n_line)

                aineq_dspijsubqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_qij_x[s_scenario][:, j_interval_window])] = np.eye(n_line)

                aineq_dspijsubqij_l[np.ix_(s_scenario*n_line*n_interval_window +
                    j_interval_window * n_line + np.arange(n_line),
                    self.var_alpha_branch_x[s_scenario])] = -np.sqrt(2) * np.diag(slmax)

        model_x_matrix_a = spar.vstack(
            [model_x_matrix_a, aineq_dspijaddqij_u, aineq_dspijaddqij_l,
             aineq_dspijsubqij_u, aineq_dspijsubqij_l])

        model_x_rhs += bineq_dspijaddqij_u.tolist() + \
                       bineq_dspijaddqij_l.tolist() + \
                       bineq_dspijsubqij_u.tolist() + \
                       bineq_dspijsubqij_l.tolist()
        model_x_senses += ['L'] * 4 * n_scenario * n_line * n_interval_window

        # ----------------------------------------------------------------------
        # Power factor
        # load_qp_ratio = (ds_sys.qload / ds_sys.pload)[:, 0]
        # (n_scenario, n_load, n_interval_window)
        load_qp_ratio = (qload_scenario / pload_scenario)[:, :, interval]
        aeq_dspowerfactor = lil_matrix(
            (n_scenario * n_load * n_interval_window, n_all_vars_x))
        beq_dspowerfactor = np.zeros(n_scenario * n_load * n_interval_window)

        for s_scenario in range(n_scenario):
            for j_interval_window in range(n_interval_window):
                aeq_dspowerfactor[np.ix_(s_scenario*n_load*n_interval_window +
                    j_interval_window * n_load + np.arange(n_load),
                    self.var_pd_x[s_scenario][:, j_interval_window])] = \
                    np.diag(load_qp_ratio[s_scenario, :, j_interval_window])

                aeq_dspowerfactor[np.ix_(s_scenario*n_load*n_interval_window +
                    j_interval_window * n_load + np.arange(n_load),
                    self.var_qd_x[s_scenario][:, j_interval_window])] = -np.eye(n_load)

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dspowerfactor])
        model_x_rhs += beq_dspowerfactor.tolist()
        model_x_senses += ['E'] * n_scenario * n_load * n_interval_window


        # nonanticipativity constraints
        # the decisions in the first interval are equal to each other in scenarios
        idx_arc_in_cut_set = [np.flatnonzero(tsn_cut_set[i_tess][:, 0])
                       for i_tess in range(n_tess)]
        n_arc_in_cut_set = [idx_arc_in_cut_set[i_tess].size for i_tess in range(n_tess)]

        n_arc_in_cut_set_all = np.array(n_arc_in_cut_set).sum()

        aeq_nonanticipativity_arc = lil_matrix(
            (n_scenario * n_arc_in_cut_set_all, n_all_vars_x))
        beq_nonanticipativity_arc = np.zeros(n_scenario * n_arc_in_cut_set_all)

        for s_scenario in range(n_scenario-1):
            for i_tess in range(n_tess):
                aeq_nonanticipativity_arc[np.ix_(
                    s_scenario*n_arc_in_cut_set_all + np.array(n_arc_in_cut_set[:i_tess]).sum()
                    + np.arange(n_arc_in_cut_set[i_tess]),
                    self.var_tess_arc_x[s_scenario][i_tess][tsn_cut_set[i_tess][:, 0]])] \
                    = np.eye(n_arc_in_cut_set[i_tess])

                aeq_nonanticipativity_arc[np.ix_(
                    s_scenario * n_arc_in_cut_set_all + np.array(
                        n_arc_in_cut_set[:i_tess]).sum()
                    + np.arange(n_arc_in_cut_set[i_tess]),
                    self.var_tess_arc_x[s_scenario+1][i_tess][
                        tsn_cut_set[i_tess][:, 0]])] \
                    = -np.eye(n_arc_in_cut_set[i_tess])


        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_nonanticipativity_arc])
        model_x_rhs += beq_nonanticipativity_arc.tolist()
        model_x_senses += ['E'] * n_scenario * n_arc_in_cut_set_all

        # ----------------------------------------------------------------------
        # nonanticipativity constraints for alpha
        aeq_nonanticipativity_alpha = lil_matrix(
            (n_scenario * n_line, n_all_vars_x))
        beq_nonanticipativity_alpha = np.zeros(n_scenario * n_line)

        for s_scenario in range(n_scenario-1):
            aeq_nonanticipativity_alpha[np.ix_(s_scenario*n_line + np.arange(n_line),
                self.var_alpha_branch_x[s_scenario])] = np.eye(n_line)

            aeq_nonanticipativity_alpha[np.ix_(s_scenario*n_line + np.arange(n_line),
                self.var_alpha_branch_x[s_scenario+1])] = -np.eye(n_line)

        model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_nonanticipativity_alpha])
        model_x_rhs += beq_nonanticipativity_alpha.tolist()
        model_x_senses += ['E'] * n_scenario * n_line

        # ----------------------------------------------------------------------
        # nonanticipativity constraints for staion_p and station_q
        aeq_nonanticipativity_station_p = lil_matrix(
            (n_scenario * n_station, n_all_vars_x))
        beq_nonanticipativity_station_p = np.zeros(n_scenario * n_station)

        aeq_nonanticipativity_station_q = lil_matrix(
            (n_scenario * n_station, n_all_vars_x))
        beq_nonanticipativity_station_q = np.zeros(n_scenario * n_station)

        for s_scenario in range(n_scenario - 1):
            for i_station in range(n_station):
                aeq_nonanticipativity_station_p[s_scenario*n_station + i_station,
                self.var_station_p_x[s_scenario][i_station, 0]] = 1

                aeq_nonanticipativity_station_p[
                    s_scenario * n_station + i_station,
                    self.var_station_p_x[s_scenario+1][i_station, 0]] = -1

                aeq_nonanticipativity_station_q[s_scenario*n_station + i_station,
                self.var_station_q_x[s_scenario][i_station, 0]] = 1

                aeq_nonanticipativity_station_q[
                    s_scenario * n_station + i_station,
                    self.var_station_q_x[s_scenario+1][i_station, 0]] = -1

        model_x_matrix_a = spar.vstack([model_x_matrix_a,
            aeq_nonanticipativity_station_p, aeq_nonanticipativity_station_q])
        model_x_rhs += beq_nonanticipativity_station_p.tolist() + \
                       beq_nonanticipativity_station_q.tolist()
        model_x_senses += ['E'] * 2 * n_scenario * n_station

        # ----------------------------------------------------------------------
        # nonanticipativity constraints for pdch and pch
        aeq_nonanticipativity_pdch = lil_matrix(
            (n_scenario * n_tess * n_station, n_all_vars_x))
        beq_nonanticipativity_pdch = np.zeros(n_scenario * n_tess * n_station)

        aeq_nonanticipativity_pch = lil_matrix(
            (n_scenario * n_tess * n_station, n_all_vars_x))
        beq_nonanticipativity_pch = np.zeros(n_scenario * n_tess * n_station)

        for s_scenario in range(n_scenario - 1):
            for i_tess in range(n_tess):
                for i_station in range(n_station):
                    aeq_nonanticipativity_pdch[
                        s_scenario*n_tess*n_station + i_tess*n_station + i_station,
                        self.var_tess2st_pdch_x[s_scenario][i_tess, i_station, 0]] = 1

                    aeq_nonanticipativity_pdch[
                        s_scenario*n_tess*n_station + i_tess*n_station + i_station,
                        self.var_tess2st_pdch_x[s_scenario+1][i_tess, i_station, 0]] = -1

                    aeq_nonanticipativity_pch[
                        s_scenario*n_tess*n_station + i_tess*n_station + i_station,
                        self.var_tess2st_pch_x[s_scenario][i_tess, i_station, 0]] = 1

                    aeq_nonanticipativity_pch[
                        s_scenario*n_tess*n_station + i_tess*n_station + i_station,
                        self.var_tess2st_pch_x[s_scenario+1][i_tess, i_station, 0]] = -1


        model_x_matrix_a = spar.vstack([model_x_matrix_a,
                                        aeq_nonanticipativity_pdch,
                                        aeq_nonanticipativity_pch])
        model_x_rhs += beq_nonanticipativity_pdch.tolist() + \
                       beq_nonanticipativity_pch.tolist()
        model_x_senses += ['E'] * 2 * n_scenario * n_tess * n_station

        # ----------------------------------------------------------------------
        # # nonanticipativity constraints for sign of dch and ch
        # aeq_nonanticipativity_sign_dch = lil_matrix(
        #     (n_scenario * n_tess, n_all_vars_x))
        # beq_nonanticipativity_sign_dch = np.zeros(n_scenario * n_tess)
        #
        # aeq_nonanticipativity_sign_ch = lil_matrix(
        #     (n_scenario * n_tess, n_all_vars_x))
        # beq_nonanticipativity_sign_ch = np.zeros(n_scenario * n_tess)
        #
        # for s_scenario in range(n_scenario - 1):
        #     for i_tess in range(n_tess):
        #         aeq_nonanticipativity_pdch[
        #             s_scenario*n_tess + i_tess,
        #             self.var_sign_dch_x[s_scenario][i_tess, 0]] = 1
        #
        #         aeq_nonanticipativity_pdch[
        #             s_scenario*n_tess + i_tess,
        #             self.var_sign_dch_x[s_scenario+1][i_tess, 0]] = -1
        #
        #         aeq_nonanticipativity_pch[
        #             s_scenario * n_tess + i_tess,
        #             self.var_sign_ch_x[s_scenario][i_tess, 0]] = 1
        #
        #         aeq_nonanticipativity_pch[
        #             s_scenario * n_tess + i_tess,
        #             self.var_sign_ch_x[s_scenario + 1][i_tess, 0]] = -1
        #
        # model_x_matrix_a = spar.vstack([model_x_matrix_a,
        #                                 aeq_nonanticipativity_sign_dch,
        #                                 aeq_nonanticipativity_sign_ch])
        # model_x_rhs += beq_nonanticipativity_sign_dch.tolist() + \
        #                beq_nonanticipativity_sign_ch.tolist()
        # model_x_senses += ['E'] * 2 * n_scenario * n_tess

        # ----------------------------------------------------------------------
        # nonanticipativity constraints for sign of pd and qd
        aeq_nonanticipativity_pd = lil_matrix(
            (n_scenario * n_load, n_all_vars_x))
        beq_nonanticipativity_pd = np.zeros(n_scenario * n_load)

        aeq_nonanticipativity_qd = lil_matrix(
            (n_scenario * n_load, n_all_vars_x))
        beq_nonanticipativity_qd = np.zeros(n_scenario * n_load)

        for s_scenario in range(n_scenario - 1):
            aeq_nonanticipativity_pd[np.ix_(
                s_scenario * n_load + np.arange(n_load),
                self.var_pd_x[s_scenario][:, 0])] = np.eye(n_load)

            aeq_nonanticipativity_pd[np.ix_(
                s_scenario * n_load + np.arange(n_load),
                self.var_pd_x[s_scenario+1][:, 0])] = -np.eye(n_load)

            aeq_nonanticipativity_qd[np.ix_(
                s_scenario * n_load + np.arange(n_load),
                self.var_qd_x[s_scenario][:, 0])] = np.eye(n_load)

            aeq_nonanticipativity_qd[np.ix_(
                s_scenario * n_load + np.arange(n_load),
                self.var_qd_x[s_scenario+1][:, 0])] = -np.eye(n_load)

        model_x_matrix_a = spar.vstack([model_x_matrix_a,
                                        aeq_nonanticipativity_pd,
                                        aeq_nonanticipativity_qd])
        model_x_rhs += beq_nonanticipativity_pd.tolist() + \
                       beq_nonanticipativity_qd.tolist()
        model_x_senses += ['E'] * 2 * n_scenario * n_load

        # For case 2 where there are stationary ESSs
        if ds_sys.case_type in ['No ESS', 'Stationary ESS']:
            aeq_stationary = lil_matrix((n_tess, n_all_vars_x))
            beq_stationary = np.zeros(n_tess)
            # indicate which microgrid (index not including station) should tess stay
            tess2microgrid = [0, 1, 3]

            for i_tess in range(n_tess):
                aeq_stationary[i_tess, self.var_tess_arc_x[i_tess]] \
                    = tsn_sys.stationary_solution[i_tess][tess2microgrid[i_tess],
                      :]
                beq_stationary[i_tess] = tsn_sys.stationary_solution[i_tess][
                                         tess2microgrid[i_tess], :].sum()

            model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_stationary])
            model_x_rhs += beq_stationary.tolist()
            model_x_senses += ['E'] * n_tess


        ''''''
        # --------------------------------------------------------------------------
        # Add constraints into Cplex model
        # model_x_matrix_a = csr_matrix(model_x_matrix_a)
        # a_rows = model_x_matrix_a.nonzero()[0].tolist()
        a_rows = model_x_matrix_a.row.tolist()
        # No computation, only query of attributes, faster than nonzero.
        # a_cols = model_x_matrix_a.nonzero()[1].tolist()
        a_cols = model_x_matrix_a.col.tolist()
        # tolist() is for 'non-integral value in input sequence'
        # a_data = model_x_matrix_a.data.tolist()
        #  model_x_matrix_a is csr_matrix matrix, a_data element needs to be
        # float
        # a_vals = model_x_matrix_a[model_x_matrix_a.nonzero()]

        # a_vals = model_x_matrix_a[model_x_matrix_a != 0]
        #  boolean mask index array, it's faster than nonzero,
        # but result is different?
        # Note that boolean mask returns 1D-array
        # a_vals = model_x_matrix_a[a_rows, a_cols].tolist()
        #  faster than boolean mask
        a_vals = model_x_matrix_a.data

        self.linear_constraints.add(rhs=model_x_rhs, senses=model_x_senses,
                                    names=['constraint{0}'.format(i)
                                           for i in range(len(model_x_rhs))])

        self.linear_constraints.set_coefficients(zip(a_rows, a_cols, a_vals))

        # self.indicator_constraints.add_batch(
        #                 lin_expr=aeq_dskvl_e,
        #                 sense=sense_dskvl_e,
        #                 rhs=beq_dskvl_e,
        #                 indvar=indicator_dskvl_e,
        #                 complemented=complemented_dskvl_e,
        #                 indtype=[self.indicator_constraints.type_.if_] *
        #                         n_line * n_interval_window)


    def query_error(self, constraint_rows):
        '''
        Query the variable names involved in specific constraints
        :param self: cplex model
        :param constraint_rows: the indices of constraints
        :return:
        '''

        # get indices and values of variables involved in infeasible constraints
        ind_var, val_var = self.linear_constraints.get_rows(
            constraint_rows).unpack()
        # get variable names
        name_var = self.variables.get_names(ind_var)

        return name_var



