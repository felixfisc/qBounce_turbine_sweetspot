import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
#import matplotlib.colors as mcolors
#import math
#import scipy

class RawTOFPoint:
    def __init__(self, 
            index, 
            N_bin, 
            s, 
            label = None,
            darkcounts = 0.0, 
            darkcounts_err = 0.0,
            delta_t = 0.0,
            delta_t_err = 0.0,
            convolution_t = 0.0,
            convolution_t_err = 0.0):
        if s < 0.001: raise ValueError("The length of the TOF tube s [mm] must be positive")
        self.index = index
        self.s = s + 8.0 # chopper padding is 8 mm
        self.label = label
        self.bin_number = index
        self.N_bin = N_bin - darkcounts
        self.N_bin_err = np.sqrt( N_bin + darkcounts_err**2 ) # Gaussian error propagation, err(N_bin) = sqrt(N_bin); poisson
        self.t = self.index * 0.1 - delta_t - convolution_t
        self.t_err = np.sqrt( (0.5*400/4000)**2 + delta_t_err**2 + convolution_t_err**2 ) # added 0.5*bin width to error
        self.N_t = self.N_bin *10 # 10 = 4000 bins / 400 ms
        self.N_t_err = self.N_bin_err *10
        if self.t > 0.000001: 
            self.v = self.s/self.t
            self.v_err = self.s/self.t**2 *self.t_err
        else:    
            self.v = float('nan')
            self.v_err = float('nan')
        self.N_v = self.N_t * self.t**2/self.s
        self.N_v_err = np.sqrt( ( self.N_t_err * self.t**2/self.s )**2 + ( self.t_err * self.N_t *2*self.t/self.s )**2 )

class RawTOFPointList(list):
    def __init__(self, 
            N_bin_list, 
            s, 
            label = None,
            darkcounts = 0.0, 
            darkcounts_err = 0.0,
            delta_t = 0.0,
            delta_t_err = 0.0,
            convolution_t = 0.0,
            convolution_t_err = 0.0):
        super().__init__()
        self.length = len(N_bin_list)
        for i in range(self.length):
            self.append(RawTOFPoint(
                index = i, 
                N_bin = N_bin_list[i], 
                s = s, 
                label = label,
                darkcounts = darkcounts, 
                darkcounts_err = darkcounts_err,
                delta_t = delta_t,
                delta_t_err = delta_t_err,
                convolution_t = convolution_t,
                convolution_t_err = convolution_t_err
            ))
        self.s = s
        self.label = label

    @property
    def bin_number(self):
        a = []
        for i in range(self.length):
            a.append(self[i].bin_number)
        return a

    @property
    def N_bin(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_bin)
        return a

    @property
    def N_bin_err(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_bin_err)
        return a

    @property
    def t(self):
        a = []
        for i in range(self.length):
            a.append(self[i].t)
        return a

    @property
    def t_err(self):
        a = []
        for i in range(self.length):
            a.append(self[i].t_err)
        return a

    @property
    def N_t(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_t)
        return a

    @property
    def N_t_err(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_t_err)
        return a

    @property
    def v(self):
        a = []
        for i in range(self.length):
            a.append(self[i].v)
        return a

    @property
    def v_err(self):
        a = []
        for i in range(self.length):
            a.append(self[i].v_err)
        return a

    @property
    def N_v(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_v)
        return a

    @property
    def N_v_err(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_v_err)
        return a

class RebinnedTOFPoint:
    def __init__(self, 
            index, 
            origin_indices_list,
            v,
            N_v,
            N_v_err, 
            s, 
            label = None,
):

        self.index = index
        self.s = s # chopper padding already included
        self.label = label
        self.origin_indices_list = origin_indices_list
        self.origin_number = len(self.origin_indices_list)
        self.v = v
        self.v_err = None
        self.N_v = N_v
        self.N_v_err = N_v_err

class RebinnedTOFPointList(list):
    def __init__(self, 
            raw_tof_point_list, 
            v_min,
            v_max,
            v_step,
            label = None):
        super().__init__()
        self.v_step = v_step
        self.length = int( (v_max-v_min)/v_step )
        self.v_min = v_min
        self.v_max = self.v_min + self.length*self.v_step
        self.s = raw_tof_point_list.s
        self.label = label

        for i in range(self.length):
            current_v = self.v_min + i*self.v_step
            current_index_list = []
            for j in range(raw_tof_point_list.length):
                if raw_tof_point_list[j].v >= current_v and raw_tof_point_list[j].v < current_v + self.v_step:
                    current_index_list.append(raw_tof_point_list[j].index)

            if len(current_index_list) < 1:
                print(f"No values to rebin from in the interval [{current_v};{current_v + self.v_step}[ m/s")
                self.append(RebinnedTOFPoint(
                    index = i, 
                    s = self.s, 
                    label = self.label,
                    origin_indices_list = current_index_list,
                    v = current_v + 0.5 * self.v_step,
                    N_v = 0,
                    N_v_err = float('nan')
                ))
            
            else:
                rebin_N = []
                weights = []
                for i in current_index_list:
                    rebin_N.append( raw_tof_point_list[i].N_v )
                    weights.append( 1 / (raw_tof_point_list[i].N_v_err**2) )
                rebin_N = np.array(rebin_N)
                weights = np.array(weights)

                self.append(RebinnedTOFPoint(
                    index = i, 
                    s = self.s, 
                    label = self.label,
                    origin_indices_list = current_index_list,
                    v = current_v + 0.5 * self.v_step,
                    N_v = np.sum( rebin_N * weights ) / np.sum(weights),
                    N_v_err = 1 / np.sqrt( np.sum(weights) )
                ))



    @property
    def origin_number(self):
        a = []
        for i in range(self.length):
            a.append(self[i].origin_number)
        return a

    @property
    def v(self):
        a = []
        for i in range(self.length):
            a.append(self[i].v)
        return a

    @property
    def v_err(self):
        a = []
        for i in range(self.length):
            a.append(self[i].v_err)
        return a

    @property
    def N_v(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_v)
        return a

    @property
    def N_v_err(self):
        a = []
        for i in range(self.length):
            a.append(self[i].N_v_err)
        return a