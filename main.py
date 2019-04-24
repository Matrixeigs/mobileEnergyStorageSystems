"""
-------------------------------------------------
   File Name：     pytess_main_4
   Description :
   Author :       yaoshuhan
   date：          19/03/19
-------------------------------------------------
   Change Activity:
                   12/11/18: Refactor the function
                   23/11/18: csee33 -> case 132
-------------------------------------------------
"""

# from archive.define_class import DistributionSystem, TransportationSystem, \
#     StationSystem, TransportableEnergyStorage, TimeSpaceNetwork, \
#     OptimizationModel, OptimizationResult

from pytess.time_system import TimeSystem
from pytess.scenario_system import ScenarioSystem
from pytess.distribution_system import DistributionSystem
from pytess.transportation_system import TransportationSystem
from pytess.station_system import StationSystem
from pytess.transportable_energy_storage import TransportableEnergyStorage
from pytess.time_space_network import TimeSpaceNetwork
from pytess.optimization_model import OptimizationModel
from pytess.optimization_result import OptimizationResult

def pytess(**kwargs):
    '''

    :param solver:
    :return:
    '''

    import numpy as np
    # from pypower.loadcase import loadcase
    from pytess.test_case import TestCase

    LARGE_SYSTEM = kwargs['large_system']

    if not LARGE_SYSTEM:
        # Initialization of test case
        tc_sys = TestCase(name="test case file")

        # Initialization of time systems
        # n_interval is extracted from load profile in ds_sys
        time_sys = TimeSystem(name='Time system')

        # Initialization of distribution systems
        # ppnet_test = pp.from_excel()
        # ppc = loadcase(case33(isrep=True))
        # ppc_joint = combine_ppc(ppc, ppc, ppc, ppc)
        ds_sys = DistributionSystem(name='Modified 33-bus test system',
                                   ppc=tc_sys.case33(isrep=True))
        ds_sys.init_load(time_sys)

        # Initialization of Sioux Falls transportation systems, 3 times the
        # distance of each road
        ts_sys = TransportationSystem(name='Sioux Falls transportation network',
                                     tsc=tc_sys.siouxfalls(), edge_factor=3)
        # Initialization of a station system including microgrids and depots
        ss_sys = StationSystem(name='Station system', ssc=tc_sys.sscase())
        # Initialization of mobile energy storage systems
        tess_sys = TransportableEnergyStorage(name='Transportable energy storage system',
                                             tessc=tc_sys.tesscase())
        # Initialization of time-space network
        tsn_sys = TimeSpaceNetwork(name='Time-space network')

    else:
        # Initialization of test case
        tc_sys = TestCase(name="test case file")

        # Initialization of time systems
        # n_interval is extracted from load profile in ds_sys
        time_sys = TimeSystem(name='Time system')

        # Initialization of distribution systems
        # ppnet_test = pp.from_excel()
        # ppc = loadcase(case33(isrep=True))
        # ppc_joint = combine_ppc(ppc, ppc, ppc, ppc)
        ds_sys = DistributionSystem(name='6 Modified 33-bus test system',
                                    ppc=tc_sys.case33_large(isrep=True))
        ds_sys.init_load(time_sys)

        # Initialization of Sioux Falls transportation systems, 3 times the
        # distance of each road
        # ts_sys = TransportationSystem(name='Sioux Falls transportation network',
        #                              tsc=tc_sys.siouxfalls(), edge_factor=3)
        ts_sys = TransportationSystem(name='Eastern Massachusetts transportation network',
                                      tsc=tc_sys.eastern_massachusetts(),
                                      edge_factor=1)
        # Initialization of a station system including microgrids and depots
        ss_sys = StationSystem(name='Station system', ssc=tc_sys.sscase_large())
        # Initialization of mobile energy storage systems
        tess_sys = TransportableEnergyStorage(
            name='Transportable energy storage system',
            tessc=tc_sys.tesscase_large())
        # Initialization of time-space network
        tsn_sys = TimeSpaceNetwork(name='Time-space network')


    # Initialization of scenario
    scenario_sys = ScenarioSystem()
    scenario_sys.gen_scenario_load(ds_sys=ds_sys)
    scenario_sys.gen_off_road(time_sys=time_sys)

    # --------------------------------------------------------------------------
    # Modify input parameters
    MW_KW = 1000

    off_road = scenario_sys.off_road
    off_line = []
    # The sef of outage lines, 1-based
    off_line_a = np.array([
        [5, 6]]) - 1

    off_line_b = np.array([
        [5, 6], [3, 23]]) - 1

    off_line_c = np.array([
        [5, 6], [3, 23]]) - 1

    off_line_d = np.array([
        [5, 6], [3, 23], [18, 33]]) - 1

    if LARGE_SYSTEM:
        off_line_e = np.array([
            [7, 8], [25, 29]]) - 1

        off_line_f = np.array([
            [10, 11]]) - 1

        off_line = np.vstack([off_line_d, off_line_b+33,
                          off_line_c+66, off_line_d+99,
                          off_line_e+132, off_line_f+165])
    else:
        off_line = np.vstack([off_line_d, off_line_b+33,
                          off_line_c+66, off_line_d+99])

    # index_off_line = [12, 36]

    # Modify transportation network edge length, update needed.
    # ts_sys.edge['length'] *= 3
    # ts_sys.update_graph(edge_factor=3)

    # but we choose the same convention as pypower. ext_grid and gen represent
    # output parameters.
    # Modify generation resource
    if not LARGE_SYSTEM:
        # 1 depot and 4 microgrids
        ss_sys.station.loc[:, 'max_p_mw':'min_q_mvar'] = ss_sys.station.loc[:,
            'max_p_mw':'min_q_mvar'].multiply(
            np.array([1, 1.8/1.6, 1, 1.8/1.6, 1]), axis=0)
    else:
        # 2 depots and 6 microgrids
        ss_sys.station.loc[:, 'max_p_mw':'min_q_mvar'] = ss_sys.station.loc[:,
            'max_p_mw':'min_q_mvar'].multiply(
            np.array([1, 1, 1.8/1.6, 1, 1.8/1.6, 1, 1.8/1.6, 1]), axis=0)

    # station energy capacity and minimum reserve
    ratio_capacity = 0.8
    ratio_reserve = 0.1

    ss_sys.station['cap_e_mwh'] = ss_sys.station['max_p_mw'] \
                       * ratio_capacity * time_sys.n_interval

    ss_sys.station['min_r_mwh'] = ss_sys.station['cap_e_mwh'] * ratio_reserve

    # --------------------------------------------------------------------------

    ds_sys.update_fault_mapping(off_line)
    ds_sys.set_optimization_case()

    ss_sys.map_station2dsts(ds_sys=ds_sys, ts_sys=ts_sys)
    ss_sys.init_localload(time_sys=time_sys, ds_sys=ds_sys)
    ss_sys.set_optimization_case(time_sys=time_sys, ds_sys=ds_sys)
    # ss_sys.find_travel_time(ts_sys=ts_sys, tess_sys=tess_sys)
    tess_sys.set_optimization_case(ds_sys=ds_sys, ss_sys=ss_sys)
    tess_sys.init_graph_from_ts(ts_sys=ts_sys)

    # tsn_sys.set_stationary_solution(ds_sys=ds_sys, ts_sys=ts_sys,
    #                                 ss_sys=ss_sys, tess_sys=tess_sys)

    # --------------------------------------------------------------------------
    # For case comparison
    # # ds_sys.case_type = 'No ESSs'
    # # ds_sys.case_type = 'Stationary ESSs'
    # # ds_sys.case_type = 'Mobile ESSs'
    ds_sys.case_type = kwargs['case_type']
    time_sys.tell_time()  # '2019-04-22_22-55'

    # For case 1 where there is no ESSs
    if ds_sys.case_type in 'No ESSs':
        tess_sys.tess_pch_u[:] = 0
        tess_sys.tess_pdch_u[:] = 0

    ############################################################################
    # Set up optimization model, non-rolling test
    # model_list = OptimizationModel()
    # model_list.add_variables(ds_sys=ds_sys, ss_sys=ss_sys,
    #                       tess_sys=tess_sys, tsn_sys=tsn_sys)
    # model_list.add_objectives(ds_sys=ds_sys, ss_sys=ss_sys,
    #                        tess_sys=tess_sys, tsn_sys=tsn_sys)
    # model_list.add_constraints(ds_sys=ds_sys, ss_sys=ss_sys,
    #                         tess_sys=tess_sys, tsn_sys=tsn_sys)
    # model_list.solve()
    # # Export model to file *.mps
    # # model_list.write('tess_model.mps')
    # model_list.sort_results(ds_sys=ds_sys, tess_sys=tess_sys, tsn_sys=tsn_sys)
    #
    # print(model_list.re_goal)

    ############################################################################
    # rolling optimization process
    ############################################################################
    model_list = []
    result_list = []
    for j_interval in range(time_sys.n_interval):
        print("j_interval:{}".format(j_interval))
        if j_interval == 22:
            a = 1

        model_list.append(OptimizationModel())
        result_list.append(OptimizationResult(name='result_list'))
        # intervals in time window
        time_sys.update_time(j_interval)
        # update transportation graph then invoked by tess.find_travel_time
        ts_sys.update_graph(off_road=[])
        # find the current location baesed on previouse action arcs except the
        # initial interval in the entire horizon
        tess_sys.find_travel_time(time_sys=time_sys, ds_sys=ds_sys,
            ts_sys=ts_sys, ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys,
            result_list=result_list)

        tsn_sys.set_tsn_model_new(time_sys=time_sys, ds_sys=ds_sys,
            ts_sys=ts_sys, ss_sys=ss_sys, tess_sys=tess_sys)

        # Update model information
        model_list[j_interval].add_variables(time_sys=time_sys, scenario_sys=scenario_sys,
            ds_sys=ds_sys, ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys)

        model_list[j_interval].add_objectives(time_sys=time_sys, scenario_sys=scenario_sys,
            ds_sys=ds_sys, ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys)

        model_list[j_interval].add_constraints(time_sys=time_sys, scenario_sys=scenario_sys,
            ds_sys=ds_sys, ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys, result_list=result_list)

        # Set Benders decomposition
        model_list[j_interval].parameters.benders.strategy.set(
            model_list[j_interval].parameters.benders.strategy.values.full)
        # Set solving time limit in seconds
        model_list[j_interval].parameters.timelimit.set(2000)

        # it doesn't work
        # url = "https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/"
        # key = "bm_api_ext_591f92c0-2a74-486f-a7c1-d7d3ada65a85"
        # Modelling assistance
        # model_list[j_interval].parameters.read.datacheck.set(2)
        # set(parameters.read.datacheck.values.off)

        # model_result_test(model=model_list[j_interval],  time_sys=time_sys, ds_sys=ds_sys, tess_sys=tess_sys, tsn_sys=tsn_sys)
        try:
            model_list[j_interval].solve()
            model_list[j_interval].solution.status[model_list[j_interval].solution.get_status()]
            res_goal = model_list[j_interval].solution.get_objective_value()
        except:
            pass
        # temp_values = np.array(model_list[j_interval].solution.get_values())
        # from utils import pickle_solution
        # pickle_solution(temp_values)

        # Export model to file *.mps, but only for model, not including solution
        # model_list[j_interval].write('test_model.mps')

        result_list[j_interval].sort_this_result_2stage(model=model_list[j_interval],
                time_sys=time_sys, scenario_sys=scenario_sys, ds_sys=ds_sys,
                ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys)
        # Print the objective value
        print(result_list[j_interval].res_goal)

    ############################################################################
    # Consolidate rolling results
    ############################################################################
    result_final = OptimizationResult(name='consolidate result')
    result_final.consolidate_result_list_2stage(result_list=result_list, time_sys=time_sys,
        scenario_sys=scenario_sys, tc_sys=tc_sys, ds_sys=ds_sys, ts_sys=ts_sys,
        ss_sys=ss_sys, tess_sys=tess_sys, tsn_sys=tsn_sys, if_draw=True)

    # --------------------------------------------------------------------------
    # Write result to file
    import os
    import pickle
    directory_name = 'results/pickle_file'
    # if not os.path.exists(pathname):
    #     os.mkdir(pathname)

    # pathlib is preferred to os.path
    import pathlib
    # exist_ok=Ture means there is no error raised up when the directory already exists.
    # otherwise, it will raise errors.
    pathlib.Path(directory_name).mkdir(parents=True, exist_ok=True)

    # file name with case_type+date&time to results
    with open(directory_name + '/' + time_sys.time_stamp + '_' + ds_sys.case_type + '.pkl',
              'wb') as fopen:
        pickle.dump((result_list, result_final, time_sys, scenario_sys, tc_sys,
                     ds_sys, ss_sys, ts_sys, tess_sys, tsn_sys), fopen)




    pass

def model_result_test(model, time_sys, ds_sys, tess_sys, tsn_sys):

    # import pickle
    #
    # with open('temp_solution/solution.pkl', 'rb') as fopen:
    #     solution = pickle.load(fopen)
    # solution = 1
    import pickle
    # file name with case_type+date&time to results
    with open('temp_solution/' + 'solution' + '.pkl',
              'rb') as fopen:
        temp_values=pickle.load(fopen)

    result = OptimizationResult(name='result_list')

    result.sort_this_result(model=model, time_sys=time_sys, ds_sys=ds_sys, ts_sys=ts_sys, tess_sys=tess_sys, tsn_sys=tsn_sys,
                            temp_values=temp_values)
    pass


def read_case_result():

    import pickle
    pathname = 'results/pickle_file'
    with open(pathname + '/2019-04-23_00-24_Mobile ESS.pkl', 'rb') as fopen:
        result_list, _, time_sys, scenario_sys, tc_sys, ds_sys, ss_sys, ts_sys, tess_sys, tsn_sys \
            = pickle.load(fopen)

    result_final = OptimizationResult(name='consolidate result')
    # Consolidate rolling results and draw figures
    result_final.consolidate_result_list_2stage(result_list=result_list, time_sys=time_sys,
        scenario_sys=scenario_sys, tc_sys=tc_sys, ds_sys=ds_sys, ss_sys=ss_sys,
        ts_sys=ts_sys, tess_sys=tess_sys, tsn_sys=tsn_sys, if_draw=True)
    # --------------------------------------------------------------------------
    print('total cost: {}\n'.format(result_final.finalres_cost_total))

    print('critical load: {}%\n'.format(
        result_final.finalres_restoration_rate_critical))

    print('normal load: {}%\n'.format(result_final.finalres_restoration_rate_normal))

    print('total load: {}%'.format(result_final.finalres_restoration_rate))

    import pathlib

    directory_name = 'results/excel'
    pathlib.Path(directory_name).mkdir(parents=True, exist_ok=True)
    with open(directory_name + '/' + time_sys.time_stamp + '_' + ds_sys.case_type + '.txt',
            'wt') as fopen:
        print('total cost: {}\n'.format(result_final.finalres_cost_total), file=fopen)

        print('critical load: {}%\n'.format(
            result_final.finalres_restoration_rate_critical), file=fopen)

        print('normal load: {}%\n'.format(
            result_final.finalres_restoration_rate_normal), file=fopen)

        print('total load: {}%'.format(result_final.finalres_restoration_rate), file=fopen)

    result_final.write_to_excel(time_sys=time_sys, ds_sys=ds_sys, ss_sys=ss_sys, tess_sys=tess_sys,
                                tsn_sys=tsn_sys)

    pass


if __name__ == '__main__':

    read_case_result()

    # model_result_test()

    # Comparative case studies
    # case_list = ['No ESS', 'Stationary ESS', 'Mobile ESS']
    case_list = ['Mobile ESS']
    LARGE_SYSTEM = False

    # Conduct three case studies in a batch
    for case_type in case_list:
        pytess(case_type=case_type, large_system=LARGE_SYSTEM)

