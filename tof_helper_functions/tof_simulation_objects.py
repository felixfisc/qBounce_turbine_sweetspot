import numpy as np
from matplotlib import pyplot as plt
import math

from tof_helper_functions.tof_framework import RawTOFPointList


class SimUCNs:
    # Only for very trivial velocity distributions, e.g. delta or square distributions
    # For more complex distributions, use the RawTOFPointList class instead 
    def __init__(self, name, num_v=10):
        self.name = name
        self.num_v = num_v
        self.data = np.zeros(num_v)

    def set_velocity(self, v_mps, N=100):
        if v_mps < 0 or v_mps >= self.num_v:
            raise ValueError("Velocity index out of bounds")
        self.data[v_mps]=N

    def plot(self):
        plt.clf()
        plt.scatter(range(self.num_v), self.data, label=self.name, color='#9DD9D2')
        plt.xlabel("v [m/s]")
        plt.ylabel("N [counts]")
        plt.title("Simulated incoming UCNs")
        plt.legend()
        plt.grid(True)
        plt.show()

class Chopper:
    def __init__(self, 
                 name, 
                 num_chop=10,
                 t_open_ms = 20,
                 t_close_ms = 380):
        self.name = name
        self.num_chop = num_chop
        self.t_open_ms = t_open_ms
        self.t_close_ms = t_close_ms
        self.cycle_time_ms = self.t_open_ms + self.t_close_ms
        self.ms_per_chop = self.cycle_time_ms/self.num_chop
        self.time_ms = np.zeros(self.num_chop)
        self.opening_value = np.zeros(self.num_chop)
        self.chopper_integral = 0

        for i in range(self.num_chop):
            self.time_ms[i]= i * self.ms_per_chop
            if i*self.ms_per_chop <= self.t_open_ms:
                self.opening_value[i] = 1.0
        

    def set_open_at(self, t_ms, fraction=1):
        if t_ms < 0 or t_ms >= (self.num_chop)*self.ms_per_chop:
            raise ValueError("Time index out of bounds")
        self.opening_value[int(t_ms/self.ms_per_chop)]=fraction

    def renormalize_opening_function(self):
        self.chopper_integral = np.sum(self.opening_value)
        for i in range(self.num_chop):
            self.opening_value[i] = self.opening_value[i] / self.chopper_integral

    def plot(self):
        plt.clf()
        plt.plot(self.time_ms, self.opening_value, label=self.name, marker ='o')
        plt.xlabel("t [ms]")
        plt.ylabel("Open")
        plt.title("Chopper opening function")
        plt.legend()
        plt.grid(True)
        plt.show()

class Detector:
    def __init__(
            self,
            num_bins = 4000,
            t_max_ms = 400,
            s_mm = 639,
            do_cycle = False,
            do_gaussian_noise = False,
            sigma = 0.0):
        self.num_bins = num_bins
        self.t_max_ms = t_max_ms
        self.s_mm = s_mm
        self.do_cycle = do_cycle
        self.do_gaussian_noise = do_gaussian_noise
        self.sigma = sigma
        self.bins_per_ms = num_bins/t_max_ms
        self.is_SimUCNs = False
        self.is_RawTOFPointList = False
        
        if self.do_gaussian_noise:
            np.random.seed(0) # for reproducibility
            self.data = [np.random.normal(0,self.sigma) for i in range(self.num_bins)]
            for a in range(self.num_bins):
                if self.data[a]<0:
                    self.data[a]=0
        else:
            self.data = np.zeros(num_bins)

    def plot(self, cut1 = None, cut2=None, title="Measured UCNs"):
        plt.clf()
        plt.scatter(range(self.num_bins)[cut1:cut2], self.data[cut1:cut2], label="TOF detector data", color='#392F5A')
        plt.xlabel("# bin")
        plt.ylabel("counts")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def measure(self, neutrons, chopper):

        if isinstance(neutrons, SimUCNs):
            self.is_SimUCNs = True
        elif isinstance(neutrons, RawTOFPointList):
            self.is_RawTOFPointList = True
        if not self.is_RawTOFPointList and not self.is_SimUCNs:
            raise TypeError("Argument 'neutrons' must be of type 'SimUCNs'")
        if not isinstance(chopper, Chopper):
            raise TypeError("Argument 'chopper' must be of type 'Chopper'")

        if self.is_SimUCNs:
            for i_chop in range(chopper.num_chop):
                for v in range(len(neutrons.data)):
                    if np.abs(v) > 0.01:
                        t_ms =  self.s_mm/v
                        t_total_ms = chopper.time_ms[i_chop] + t_ms
                        bin = math.floor(self.bins_per_ms*t_total_ms)
                        if self.do_cycle == True:
                            self.data[bin % self.num_bins]+=chopper.opening_value[i_chop]*neutrons.data[v]
                        else:
                            if bin < self.num_bins:
                                self.data[bin]+=chopper.opening_value[i_chop]*neutrons.data[v]

        if self.is_RawTOFPointList:
            if self.s_mm != neutrons.s:
                print(f"Warning: Detector's s_mm = {self.s_mm} while neutrons' s = {neutrons.s} mm.")
            for i_chop in range(chopper.num_chop):
                for i in range(neutrons.length):
                    if np.abs(neutrons[i].v) > 0.01:
                        t_ms =  self.s_mm/neutrons[i].v
                        t_total_ms = chopper.time_ms[i_chop] + t_ms
                        bin = math.floor(self.bins_per_ms*t_total_ms)
                        if self.do_cycle == True:
                            self.data[bin % self.num_bins]+=chopper.opening_value[i_chop]*neutrons[i].N_bin
                        else:
                            if bin < self.num_bins:
                                self.data[bin]+=chopper.opening_value[i_chop]*neutrons[i].N_bin
