"""
-------------------------------------------------
   File Name：     init_load
   Description :
   Author :       yaoshuhan
   date：          12/11/18
-------------------------------------------------
   Change Activity:
                   12/11/18:
-------------------------------------------------
"""
import numpy as np
from numpy import array, arange, setdiff1d, hstack, zeros, isin, interp, \
    nonzero, tile, unique


def init_load_type_cost(ppnet):
    '''
    Initialization of load category and interruption cost
    :param ppnet:
    :return:
    '''

    bus =ppnet.bus
    load = ppnet.load
    n_bus = bus.shape[0]

    # Load type
    # each category of loads' bus no.
    load_category = dict()
    # bus number, convert to 1-base to 0-base
    # for 33-ubs
    # load_category['industrial'] = array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    #                                      33]) - 1
    # load_category['commercial'] = array([9, 10, 11, 12, 13, 14, 15, 16,
    #                                      17, 18]) - 1
    # load_category['residential'] = setdiff1d(arange(n_bus), hstack(
    #     (load_category['industrial'], load_category['commercial'])))

    # for 132 bus
    load_category['industrial'] = arange(66, 99)
    load_category['commercial'] = np.hstack([arange(33, 66), arange(99, 132)])
    load_category['residential'] = setdiff1d(arange(n_bus), hstack(
        (load_category['industrial'], load_category['commercial'])))

    # this would include the substation bus with no load, but this would be
    # excluded by isin on load['bus']

    # Test whether each element of a 1-D array ar1 is also present in a second
    # array ar2.
    for e_load_type in load_category.keys():
        load.loc[isin(load['bus'], load_category[e_load_type]),
                 'load_type'] = e_load_type

    # Load bus # with priority levels
    load_priority = dict()
    # load_priority['A'] = array([2, 4, 7, 8, 14, 24, 25, 29, 30, 31, 32, 15,
    #                             16, 20, 22]) - 1
    # set seed to generate the same random values each time
    # np.random.seed(2018)
    # load_priority['A'] = np.random.randint(low=0, high=n_bus-1, size=62)
    # Original data
    a_is_more = np.array([2, 4, 7, 8, 14, 18, 19, 20,
                          24, 25, 29, 30, 31, 32]) - 1
    a_is_less = np.array([3, 5, 6, 9, 10, 12, 13, 15, 16, 17,
                          23, 28]) - 1

    from functools import reduce
    load_priority['A'] = reduce(np.union1d, (a_is_less+0, a_is_more+33,
                                             a_is_less+66, a_is_more+99))
    load_priority['B'] = array([])
    load_priority['C'] = setdiff1d(arange(n_bus),
                            hstack((load_priority['A'], load_priority['B'])))

    # load cost for each priority level
    load_cost = dict()
    # $ / MWh
    load_cost['A'] = 10 * 1000
    load_cost['B'] = 2 * 1000
    load_cost['C'] = 2 * 1000

    # Test whether each element of a 1-D array ar1 is also present in a second
    # array ar2.
    for e_load_priority in load_priority:
        load.loc[isin(load['bus'], load_priority[e_load_priority]),
                 'load_cost'] = load_cost[e_load_priority]

    return ppnet


def init_load_profile(load, time_sys, ds_sys, LOAD_TYPE='load_type',
                      P_LOAD='p_mw', Q_LOAD='q_mvar'):
    '''
    To initialize load profile
    :param load:
    :param LOAD_TYPE:
    :return:
    '''

    n_load = load.shape[0]

    n_interval = time_sys.n_interval

    # To generate load profile
    index_interval = arange(n_interval)
    # index_interval = arange(24)
    load_profile_reference = {}

    # Industiral load profile (original data)
    industrial_load = zeros((2, 24))
    industrial_load[0, :] = arange(24)
    # Original load
    # industrial_load[1, :] = array([50, 38, 27, 27, 26, 24, 22, 42,
    #                                45, 60, 52, 50, 50, 48, 58, 50,
    #                                62, 57, 82, 84, 96, 100, 87, 60])
    # modified
    industrial_load[1, :] = array([50, 45, 50, 53, 57, 54, 52, 54,
                                   55, 60, 52, 50, 50, 48, 58, 50,
                                   62, 57, 82, 84, 96, 100, 87, 60])

    industrial_load = interp(index_interval, industrial_load[0, :],
                             industrial_load[1, :]) / 100
    industrial_load = industrial_load * 1
    load_profile_reference['industrial'] = industrial_load  # (n_interval,)

    # Commercial load profile (original data)
    commercial_load = zeros((2, 24))
    commercial_load[0, :] = arange(24)
    commercial_load[1, :] = array([28, 27, 28, 27, 29, 30, 35, 70,
                                   85, 96, 96, 96, 90, 84, 80, 60,
                                   50, 45, 40, 40, 40, 40, 40, 35])
    commercial_load = interp(index_interval, commercial_load[0, :],
                             commercial_load[1, :]) / 100
    commercial_load = commercial_load * 1
    load_profile_reference['commercial'] = commercial_load  # (n_interval,)

    # Residential load profile (original data)
    residential_load = zeros((2, 24))
    residential_load[0, :] = arange(24)
    residential_load[1, :] = array([32, 30, 28, 26, 26, 25, 25, 25,
                                    22, 23, 75, 82, 88, 78, 55, 40,
                                    65, 80, 95, 100, 90, 65, 45, 35])
    residential_load = interp(index_interval, residential_load[0, :],
                              residential_load[1, :]) / 100
    load_profile_reference['residential'] = residential_load  # (n_interval,)

    # Generate load profile ----------------------------------------------------
    pload_profile = zeros((n_load, n_interval))
    qload_profile = zeros((n_load, n_interval))

    for e_load_type in ['industrial', 'commercial', 'residential']:
        n_e_load_type = (load[LOAD_TYPE] == e_load_type).nonzero()[
            0].size  # the number of loads for each type
        # load profile
        pload_profile[load[LOAD_TYPE] == e_load_type, :] = tile(
            load_profile_reference[e_load_type], reps=(n_e_load_type, 1))

        qload_profile[load[LOAD_TYPE] == e_load_type, :] = tile(
            load_profile_reference[e_load_type], reps=(n_e_load_type, 1))

    # Obtain load (n_load, n_interval)
    pload = load[P_LOAD][:, np.newaxis] * pload_profile / ds_sys.sn_mva
    qload = load[Q_LOAD][:, np.newaxis] * qload_profile / ds_sys.sn_mva

    return pload, qload, load_profile_reference


def get_load_info(ppnet, BUS_I='bus', LOAD_COST='load_cost',
                  LOAD_TYPE='load_type'):
    """
    :param ppnet: pand
    :param BUS_I:
    :param LOAD_COST:
    :param LOAD_TYPE:
    :return: load_information: dictionary of arrays, each item refers to each
    type of load and reveal load bus number and load cost
    """

    load = ppnet.load
    load_information = {}

    # n_load_type = unique(bus[LOAD_TYPE]).shape[0]
    # The prefix e means element
    # for e_load_type in unique(load[LOAD_TYPE]):
    #     index_load = nonzero(bus[LOAD_TYPE] == e_load_type)[0]
    #     load_information.append(bus.loc[index_load, [BUS_I, LOAD_COST]])

    for e_load_type in unique(load[LOAD_TYPE]):
        load_information[e_load_type] = load.loc[
            load['load_type'] == e_load_type, [BUS_I, LOAD_COST]]

    return load_information