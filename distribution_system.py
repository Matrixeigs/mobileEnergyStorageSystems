import numpy as np
from numpy import array, zeros, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import cplex as cpx

class DistributionSystem:
    '''

    '''

    def __init__(self, name, ppc):

        import pandapower.converter as pc

        self.name = name
        # create pandapower net from ppc
        self.ppnet = pc.from_ppc(ppc)

        # shortener, note it is view other than copy.
        bus = self.ppnet.bus
        line = self.ppnet.line
        ext_grid = self.ppnet.ext_grid
        load = self.ppnet.load
        gen = self.ppnet.gen

        # since line will be modified, store the original line, load
        self.origin_line = line

        # Get bus index lists of each type of bus
        self.idx_ref_bus = np.array(ext_grid['bus'], dtype=int)
        self.idx_pv_bus = np.array(gen['bus'], dtype=int)
        self.idx_pq_bus = np.setdiff1d(np.array(bus.index, dtype=int),
                               np.hstack((self.idx_ref_bus, self.idx_pv_bus)))

        # The no. of bus, ... in distribution system
        self.n_bus = bus.shape[0]
        self.n_line = line.shape[0]
        self.n_ext_grid = ext_grid.shape[0]
        self.n_load = load.shape[0]

        # Base value for S capacity
        self.sn_mva = self.ppnet.sn_mva
        self.vn_kv = bus.loc[0, 'vn_kv']
        # Base value for current
        self.in_ka = self.sn_mva / (np.sqrt(3) * self.vn_kv)

        # branch resistance and reactance
        # check if there are transformers?
        # if no trafo:
        line['branch_r_pu'] = (line['r_ohm_per_km'] * line['length_km']
                            / line['parallel'] /
                        ((self.vn_kv * 1e3) ** 2 / (self.sn_mva * 1e6)))
        line['branch_x_pu'] = (line['x_ohm_per_km'] * line['length_km']
                            / line['parallel'] /
                        ((self.vn_kv * 1e3) ** 2 / (self.sn_mva * 1e6)))

        # self.init_load()
        # Find all the immediately neighboring nodes connecting to each MG bus
        # The index for branches starting from MG bus
        # self.idx_beta_ij = line['from_bus'].index[
        #     np.isin(line['from_bus'], self.idx_ref_bus)].values
        #
        # # self.idx_beta_ij = array()
        # # The index for branches ending at MG bus
        # self.idx_beta_ji = line['to_bus'].index[
        #     np.isin(line['to_bus'], self.idx_ref_bus)].values


    def init_load(self, time_sys):
        '''

        :return:
        '''

        from pytess.load_info import init_load_type_cost, init_load_profile, \
            get_load_info

        # generate load type, load interruption cost and load profile
        self.ppnet = init_load_type_cost(self.ppnet)

        self.pload, self.qload, self.load_profile_reference = init_load_profile(load=self.ppnet.load,
            time_sys=time_sys, ds_sys=self)

        # consolidate load information
        self.load_information = get_load_info(self.ppnet)

    def update_fault_mapping(self, off_line=[]):
        '''
        Update distribution system considering outage lines and buses
        :param idx_off_line:
        :return:
        '''

        import copy
        import pandapower as pp
        import pandapower.topology as ptop
        import numpy_indexed as npi

        bus = self.ppnet.bus
        ext_grid = self.ppnet.ext_grid
        gen = self.ppnet.gen
        load = self.ppnet.load
        # the updated line should be a copy other than a view
        line = copy.deepcopy(self.origin_line)
        # view to self.line
        self.ppnet.line = line

        # Find faults and update self.ppnet
        line['in_service'] = True
        bus['in_service'] = True

        # Find out index of outage lines
        all_line = self.ppnet.line.loc[:,
                   'from_bus': 'to_bus'].values.astype(int)

        temp = npi.intersection(off_line, all_line)
        idx_off_line = np.sort(npi.indices(all_line, temp))

        line.loc[idx_off_line, 'in_service'] = False
        # Identify areas and remove isolated ones -------------------------------------------------------------------------
        # Set all isolated buses and all elements connected to isolated buses
        # out of service.
        # Before this, it needs to add microgrid to ext_grid
        pp.set_isolated_areas_out_of_service(self.ppnet)  # isolated means that
        # a bus is disconnected from ext_grid
        idx_off_bus = bus[bus['in_service'] == False].index.values
        idx_off_line = line[line['in_service'] == False].index.values

        # Remove all the faulty lines and update no. of lines
        line.drop(labels=idx_off_line, inplace=True)
        # Update the no. of lines
        self.n_line = line.shape[0]
        # Reset line index starting at zero
        line.reset_index(drop=True, inplace=True)

        # Mask outage load in mapping of load to distribution bus
        idx_off_load = np.flatnonzero(np.isin(load['bus'], idx_off_bus))
        idx_on_load = np.setdiff1d(load.index, idx_off_load)
        n_on_load = idx_on_load.size  # no. of on load
        # the mask of load on/off status, (n_load, 1)
        self.on_off_load = np.isin(load.index, idx_on_load)
        # update mapping function of load to distribution bus
        self.mapping_load2dsbus = csr_matrix(
            (ones(n_on_load), (idx_on_load, load.loc[idx_on_load, 'bus'])),
            shape=(self.n_load, self.n_bus), dtype=int)

        # set distribution system graph
        self.ds_graph = ptop.create_nxgraph(self.ppnet)

        # # todo update index lists of each type of bus
        # self.idx_ref_bus = np.array(ext_grid['bus'], dtype=int)
        # self.idx_pv_bus = np.array(gen['bus'], dtype=int)
        # self.idx_pq_bus = np.setdiff1d(np.array(bus.index, dtype=int),
        #                        np.hstack((self.idx_ref_bus, self.idx_pv_bus)))

        # Find all the immediately neighboring nodes connecting to each MG bus
        # The index for branches starting from MG bus
        # self.idx_beta_ij = line['from_bus'].index[
        #     np.isin(line['from_bus'], self.idx_ref_bus)].values
        # # The index for branches ending at MG bus
        # self.idx_beta_ji = line['to_bus'].index[
        #     np.isin(line['to_bus'], self.idx_ref_bus)].values

        # set up new incidence matrix of distribution bus to lines
        # from_bus to line
        self.incidence_ds_fbus2line = csr_matrix(
            (ones(self.n_line), (range(self.n_line), line['from_bus'])),
            shape=(self.n_line, self.n_bus), dtype=int).T
        # to_bus to line
        self.incidence_ds_tbus2line = csr_matrix(
            (ones(self.n_line), (range(self.n_line), line['to_bus'])),
            shape=(self.n_line, self.n_bus), dtype=int).T

    def set_optimization_case(self):
        '''

        :return:
        '''
        line = self.ppnet.line
        bus = self.ppnet.bus
        ext_grid = self.ppnet.ext_grid
        load = self.ppnet.load

        # load interruption cost, (n_load, 1)
        self.load_interruption_cost = load['load_cost'][:, np.newaxis]

        # parameters for line capacity limit, (n_line, 1)
        self.slmax = line['max_i_ka'].values[:, np.newaxis] / self.in_ka

        # Upper bounds
        # line capacity for active power at each interval, (n_line, 1)
        self.pij_u = self.slmax[:, np.newaxis]
        # line capacity for reactive power,
        # (n_line, 1)
        self.qij_u = self.slmax[:, np.newaxis]
        # line capacity for apparent power at each interval,
        # (n_line, 1)
        self.sij_u = self.slmax[:, np.newaxis]
        # bus voltage
        # v0 is for bus voltage constant
        self.v0 = 1
        self.vm_u = bus['max_vm_pu'][:, np.newaxis]
        self.vm_u[ext_grid['bus'].astype(int), :] = self.v0

        # Lower bounds
        # since the power flow on line is bidirectional
        self.pij_l = -self.slmax[:, np.newaxis]
        self.qij_l = -self.slmax[:, np.newaxis]
        # bus voltage
        self.vm_l = bus['min_vm_pu'][:, np.newaxis]
        self.vm_l[ext_grid['bus'].astype(int), :] = self.v0