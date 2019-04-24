"""
-------------------------------------------------
   File Name：     optimization_model_toolbox
   Description :
   Author :       yaoshuhan
   date：          19/11/18
-------------------------------------------------
   Change Activity:
                   19/11/18:
-------------------------------------------------
"""


import numpy as np
from numpy import array
import cplex as cpx

def draw_tsn_route(result, tess_sys, tsn_sys):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    #for seaborn style
    sns.set()

    # The # of columns of subplots
    n_col_subplot = 2
    tsn_fig, tsn_axes = plt.subplots(np.ceil(tess_sys.n_tess /
                                             n_col_subplot).astype(int), n_col_subplot)

    result.res_route_coordinate = []
    # For each tess corresponding to a subplot
    for i_tess in range(tess_sys.n_tess):
        # Parameters to for scatter plot, including all time-space nodes
        node_rows, node_cols = np.nonzero(tsn_sys.tsn_node[i_tess, :, :] >= 0)
        node_strs = tsn_sys.tsn_node[i_tess, node_rows, node_cols]
        node_x_axis, node_y_axis = node_cols, node_rows

        # The position of subplot
        subplot_row, subplot_col = np.divmod(i_tess, n_col_subplot)

        # Scatter plot for time-space arc_nodes
        tsn_axes[subplot_row, subplot_col].plot(node_x_axis,
                                                        node_y_axis, 's')
        tsn_axes[subplot_row, subplot_col].set(xlabel='Time (h)',
                                                       ylabel='Station (#)',
                                    title='TESS #{0} routes'.format(i_tess))
        # Each time-space node has a tag, it can only use for loop to assign.
        for e_x, e_y, e_str in zip(node_x_axis, node_y_axis, node_strs):
            tsn_axes[subplot_row, subplot_col].text(x=e_x, y=e_y, s=e_str)

        # Line plot for time-space arcs
        # for i_tess_route in result.res_tess_route_x:
        i_tess_route = result.res_tess_route_x[i_tess]
            # identify all the involving arc_nodes
        # Only includes nodes of the active arcs
        arc_nodes = np.union1d(i_tess_route[:, 0], i_tess_route[:, 1])
        # To get each time-space node's coordinates
        arc_node_coordinates = np.vstack(np.nonzero(
            np.isin(tsn_sys.tsn_node[i_tess, :, :], arc_nodes)))
        # need to sort
        # arc_node_coordinates
        # = arc_node_coordinates[:, np.lexsort(arc_node_coordinates)]
        # a.T[np.lexsort(a)].T, why does it work?
        arc_node_coordinates = arc_node_coordinates.T[
            np.lexsort(arc_node_coordinates)].T
        # Swap two rows so that the first row is x_axis, the second row
        # is y_axis
        arc_node_coordinates[[0, 1], :] = arc_node_coordinates[[1, 0], :]
        tsn_axes[subplot_row, subplot_col].plot(arc_node_coordinates[0],
                                                arc_node_coordinates[1])

        result.res_route_coordinate.append(arc_node_coordinates)


def multiple_df2single_sheet(df_list, sheet_name, writer, space):
    '''
    Write multiple DataFrame into one xlsx sheet
    use: multiple_dfs(dfs, 'Validation', 'test1.xlsx', 1)
    :param df_list:
    :param sheet_name:
    :param writer:
    :param space:
    :return:
    '''
    # start row
    row = 0
    # Write multiple dataframe into one sheet in sequence
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheet_name, startrow=row,
                           startcol=0)
        # The start row for next df
        row = row + len(dataframe.index) + space + 1

def draw_figure(result, ds_sys, ss_sys, tess_sys, tsn_sys):

    import matplotlib.pyplot as plt

    # draw the route on tsn graph
    # draw_tsn_route(result=result, tess_sys=tess_sys, tsn_sys=tsn_sys)

    tess2station_fig, tess2station_axes = plt.subplots(3)
    tess2station_axes[0].plot(result.finalres_tess_netoutp_x.T)
    tess2station_axes[0].set(title='tess output power')
    # tess2station_axes[1].plot(result.res_station_netinp_x.T)
    # tess2station_axes[1].set(title='')
    tess2station_axes[1].plot(result.finalres_tess_soc_x.T)
    tess2station_axes[1].set(title='tess soc')
    tess2station_axes[2].plot(result.finalres_e_transfer_station.T)
    tess2station_axes[2].set(title='energy transfer among stations')

    station_fig, station_axes = plt.subplots(3)
    station_axes[0].plot(result.finalres_station_p_x.T)
    station_axes[1].plot(result.finalres_station_q_x.T)
    # station_axes[2].plot(result.res_station_e_x.T)

    load_fig, load_axes = plt.subplots(2)
    load_axes[0].plot(result.finalres_pd_x.T)
    load_axes[1].plot(result.finalres_qd_x.T)

    # plt.ion()
    plt.show()

    pass

def pickle_solution(solution):
    # Write result to file
    import pickle
    # file name with case_type+date&time to results
    with open('temp_solution/' + 'solution' + '.pkl',
              'wb') as fopen:
        pickle.dump(solution, fopen)
