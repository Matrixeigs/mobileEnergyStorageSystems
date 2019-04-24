

import numpy as np
import matplotlib.pyplot as plt

class ScenarioSystem():
    def __init__(self):
        self.name = "Scenario"
        self.n_scenario = 5

    def gen_scenario_load(self, ds_sys, LOAD_TYPE='load_type',
                          P_LOAD='p_mw', Q_LOAD='q_mvar'):
        # # (n_load, n_interval)
        # pload_base_value = ds_sys.pload
        # qload_base_value = ds_sys.qload
        #
        # n_scenario = 1000
        #
        # # (n_scenario, n_load, n_interval)
        # pload_scenario = np.tile(pload_base_value[np.newaxis, ...], (n_scenario, 1, 1))
        # qload_scenario = np.tile(qload_base_value[np.newaxis, ...], (n_scenario, 1, 1))
        #
        # np.random.seed(188)
        # pload_realization = np.random.normal(pload_base_value, pload_base_value * 0.02)
        # qload_realization = np.random.normal(qload_base_value, qload_base_value * 0.02)
        #
        # pload_scenario = np.random.normal(pload_scenario, pload_scenario * 0.02)
        # qload_scenario = np.random.normal(qload_scenario, qload_scenario * 0.02)

        n_scenario = self.n_scenario

        # a dict of (n_interval, )
        load_profile_base_value = ds_sys.load_profile_reference
        n_interval = load_profile_base_value['industrial'].shape[0]
        load_keys = load_profile_base_value.keys()  # ['industrial', 'commercial', 'residential']

        # np.random.seed(188)
        # load_profile_realization = dict()
        # for i_key in load_keys:
        #     # (n_interval, )
        #     mu = load_profile_base_value[i_key]
        #     sigma = mu * 0.02
        #     load_profile_realization[i_key] = np.random.normal(mu, sigma)
        #
        # import matplotlib.pyplot as plt
        #
        # plt.plot(load_profile_base_value['commercial'])
        # plt.plot(load_profile_realization['commercial'])
        #
        # plt.show()
        np.random.seed(188)
        load_profile_scenario = dict()
        # For prediction interval
        load_profile_scenario_max = dict()
        load_profile_scenario_min = dict()
        load_profile_realization = dict()
        rand_max_ratio = np.random.rand(n_interval)  # random number in [0, 1)
        for i_key in load_keys:
            mu = np.tile(load_profile_base_value[i_key][np.newaxis, :], (n_scenario, 1))
            sigma = mu * 0.02
            load_profile_scenario[i_key] = np.random.normal(mu, sigma)
            # load_profile_scenario_max[i_key] = load_profile_scenario[i_key].max(axis=0)
            # load_profile_scenario_min[i_key] = load_profile_scenario[i_key].min(axis=0)
            load_profile_scenario_max[i_key] = load_profile_base_value[i_key] + \
                                               2 * 0.02 * load_profile_base_value[i_key]
            load_profile_scenario_min[i_key] = load_profile_base_value[i_key] - \
                                               2 * 0.02 * load_profile_base_value[i_key]
            load_profile_realization[i_key] = load_profile_scenario_max[i_key] * rand_max_ratio + \
                                              load_profile_scenario_min[i_key] * (1 - rand_max_ratio)

        # Write to excel for use in originPro
        # write_to_excel(load_keys, load_profile_base_value, load_profile_scenario_min, load_profile_scenario_max,
        #                load_profile_realization)

        # For scenario reduction
        load_profile_scenario_reduction = load_profile_scenario
        scenario_weight = np.ones(n_scenario) / n_scenario
        # scenario_array = np.zeros((n_scenario, 0))
        # for i_key in load_keys:
        #     i_array = load_profile_scenario[i_key]
        #     scenario_array = np.hstack((scenario_array, i_array))
        #
        # weight = np.ones(n_scenario) / n_scenario
        # n_reduced = n_scenario - 5
        # power = 2
        # scenario_reduction = ScenarioReduction()
        #
        # scenario_reduced, weight_reduced = scenario_reduction.run(
        #     scenario=scenario_array, weight=weight, n_reduced=n_reduced, power=power)

        # Generate load profile ----------------------------------------------------
        load = ds_sys.ppnet.load
        n_load = load.shape[0]
        # pload_profile_aug is the load multiplier for each loads in ds
        pload_multiplier_realization = np.zeros((n_load, n_interval))
        qload_multiplier_realization = np.zeros((n_load, n_interval))

        pload_profile_aug = np.zeros((n_scenario, n_load, n_interval))
        qload_profile_aug = np.zeros((n_scenario, n_load, n_interval))


        for i_key in load_keys:
            n_load_this_load_type = (load[LOAD_TYPE] == i_key).nonzero()[
                0].size  # the number of loads for each type
            # load profile
            pload_multiplier_realization[load[LOAD_TYPE] == i_key, :] = np.tile(
                load_profile_realization[i_key][np.newaxis, :], reps=(n_load_this_load_type, 1)
            )

            qload_multiplier_realization[load[LOAD_TYPE] == i_key, :] = np.tile(
                load_profile_realization[i_key][np.newaxis, :], reps=(n_load_this_load_type, 1)
            )

            pload_profile_aug[:, load[LOAD_TYPE] == i_key, :] = np.tile(
                load_profile_scenario_reduction[i_key][:, np.newaxis, :], reps=(1, n_load_this_load_type, 1))

            qload_profile_aug[:, load[LOAD_TYPE] == i_key, :] = np.tile(
                load_profile_scenario_reduction[i_key][:, np.newaxis, :], reps=(1, n_load_this_load_type, 1))

        # Obtain load (n_load, n_interval)
        pload_realization = load[P_LOAD][:, np.newaxis] * pload_multiplier_realization / ds_sys.sn_mva
        qload_realization = load[Q_LOAD][:, np.newaxis] * qload_multiplier_realization / ds_sys.sn_mva

        pload_scenario = load[P_LOAD][np.newaxis, :, np.newaxis] * pload_profile_aug / ds_sys.sn_mva
        qload_scenario = load[Q_LOAD][np.newaxis, :, np.newaxis] * qload_profile_aug / ds_sys.sn_mva

        # a = np.average(pload_scenario, axis=0)

        assert pload_realization.shape == (n_load, n_interval) and \
               qload_realization.shape == (n_load, n_interval) and \
               pload_scenario.shape == (n_scenario, n_load, n_interval) and \
               qload_scenario.shape == (n_scenario, n_load, n_interval)

        # import matplotlib.pyplot as plt
        # plt.plot(load_profile_base_value['industrial'])
        # plt.plot(load_profile_scenario_max['industrial'])
        # plt.plot(load_profile_scenario_min['industrial'])

        # Test
        # mu = pload_base_value[8, 16]
        # sigma = mu * 0.02
        # count, bins, ignored = plt.hist(pload_scenario[:, 8, 16], 100, density=True)
        # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
        #     np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
        #     linewidth = 2, color = 'r')
        # plt.show()

        pass

        self.scenario_weight = scenario_weight

        self.load_profile_base_value = load_profile_base_value
        self.load_profile_scenario_min = load_profile_scenario_min
        self.load_profile_scenario_max = load_profile_scenario_max
        self.load_profile_realization = load_profile_realization


        self.pload_realization = pload_realization
        self.qload_realization = qload_realization
        self.pload_scenario = pload_scenario
        self.qload_scenario = qload_scenario

        return pload_realization, qload_realization, pload_scenario, qload_scenario

    def gen_off_road(self, time_sys):
        # off_road is a list of tuple
        # off_road = [(1, 2), (2, 3)]
        # the road is represented by original node, (no minus 1), interval_start, interval_end)
        n_interval = time_sys.n_interval

        off_road_table = [
            [(9, 10), 15, 23],
            [(12, 13), 12, 18],
            [(15, 22), 4, 13]
        ]

        off_road = []

        for j_interval in range(n_interval):
            off_road_this_interval = []
            for i_fault_road, i_interval_start, i_interval_end in off_road_table:
                if (j_interval >= i_interval_start) and (j_interval <= i_interval_end):
                    off_road_this_interval += [i_fault_road]

            off_road.append(off_road_this_interval)

        self.off_road = off_road

"""
Scenario reduction algorithm for two-stage stochastic programmings
The fast forward selection algorithm is used.

References:
    [1]https://edoc.hu-berlin.de/bitstream/handle/18452/3285/8.pdf?sequence=1
    [2]http://ftp.gamsworld.org/presentations/present_IEEE03.pdf
Considering the second stage optimization problem is linear programming, the distance function is refined to
c(ξ, ˜ξ) := max{1, kξkp−1, k˜ξkp−1}kξ − ˜ξk (p = 2, which is sufficient for right hand side uncertainties)

"""

def write_to_excel(load_keys, load_profile_base_value, load_profile_scenario_min, load_profile_scenario_max,
                   load_profile_realization):

    import pandas as pd
    from utils import multiple_df2single_sheet

    # Initialization of writer
    writer = pd.ExcelWriter('./case/case for originpro.xlsx',
                            engine='xlsxwriter')

    # write
    df_list = []
    for i_key in load_keys:
        table = np.vstack((load_profile_base_value[i_key],
            load_profile_scenario_min[i_key], load_profile_scenario_max[i_key],
                           load_profile_realization[i_key]))
        df_list.append(pd.DataFrame(table,
                                    index=['base value', 'min', 'max', 'realization']))

    multiple_df2single_sheet(df_list=df_list, sheet_name='load profile',
                             writer=writer, space=2)

    # Save to excel file
    writer.save()


class ScenarioReduction():
    def __init__(self):
        self.name = "Scenario reduction"

    def run(self, scenario, weight, n_reduced, power):
        """

        :param scenario: A fan scenario tree, when more stage are considered, some merge operation can be implemented
        :param weight: Weight of each scenario
        :param n_reduced: Number of scenarios needs to be reduced
        :param power: The power in the distance calculation
        :return:
        """
        from numpy import array, zeros, argmin, random, arange, linalg, ones, \
            inf, delete, where, append

        n_scenario = scenario.shape[0]  # number of original scenarios
        c = zeros((n_scenario, n_scenario))
        # Calculate the c matrix
        for i in range(n_scenario):
            for j in range(n_scenario):
                c[i, j] = linalg.norm((scenario[i, :] - scenario[j, :]), 2)
                c[i, j] = max([1, linalg.norm(scenario[i, :], power - 1), linalg.norm(scenario[j, :], power - 1)]) * \
                          c[i, j]

        J = arange(n_scenario)  # The original index range
        J_reduced = array([])
        # Implement the iteration
        for n in range(n_reduced):  # find the minimal distance
            c_n = inf * ones(n_scenario)
            c_n[J] = 0
            for u in J:
                # Delete the i-th distance
                J_temp = delete(J, where(J == u))
                for k in J_temp:
                    c_k_j = delete(c[int(k)], J_temp)
                    c_n[int(u)] += weight[int(k)] * min(c_k_j)
            u_i = argmin(c_n)
            J_reduced = append(J_reduced, u_i)
            J = delete(J, where(J == u_i))
        # Optimal redistribution
        p_s = weight.copy()
        p_s[J_reduced.astype(int)] = 0

        for i in J_reduced:
            c_temp = c[int(i), :]
            c_temp[J_reduced.astype(int)] = inf
            index = argmin(c_temp)
            p_s[index] += weight[int(i)]

        scenario_reduced = scenario[J.astype(int), :]
        weight_reduced = p_s[J.astype(int)]

        return scenario_reduced, weight_reduced


if __name__ == "__main__":
    n_scenario = 100
    scenario = np.random.random((n_scenario, 10))
    weight = np.ones(n_scenario) / n_scenario
    n_reduced = int(n_scenario / 2)
    power = 2
    scenario_reduction = ScenarioReduction()

    (scenario_reduced, weight_reduced) = scenario_reduction.run(
        scenario=scenario, weight=weight, n_reduced=n_reduced, power=power)

    print(scenario_reduced)


