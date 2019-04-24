import numpy as np


class TimeSystem:

    def __init__(self, name):
        self.name = name
        self.delta_t = 1
        # length of entire horizon
        self.n_interval = 24
        self.n_timepoint = self.n_interval + 1

        # Time window
        # set length of time window
        self.n_interval_window_set = 12
        # The optimization will be implemented for the interval
        self.n_interval_window_effect = 1


    def update_time(self, j_interval):
        """

        :param j_interval: the current time interval
        :return:
        """
        # when it comes to the later part, the length of time window may not be
        # the same with set one.
        # the actual length of the time window
        self.n_interval_window = min(self.n_interval_window_set,
                                       self.n_interval - j_interval)
        # the number of time points in the time window
        self.n_timepoint_window = self.n_interval_window + 1

        # current window involves the interval in entire horizon,
        self.interval = np.arange(j_interval, j_interval + self.n_interval_window)

        # current window involves time points in entire horizon
        self.timepoint = np.arange(j_interval, j_interval + self.n_timepoint_window)

        # inner interval index in the window
        self.interval_window = np.arange(self.n_interval_window)
        self.timepoint_window = np.arange(self.n_timepoint_window)

    def tell_time(self):
        import time
        # Obtain the debug time for naming files later to to include three cases
        # in a time batch
        self.time_stamp = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
