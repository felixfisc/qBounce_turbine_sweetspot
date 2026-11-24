import numpy as np
import scipy
import matplotlib.pyplot as plt

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
        self.s = s + 8.0 # chopper padding is 8 mm
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
        self.v_err = 0.0 # set to zero because v is a theory quantity without an uncertainty, left as a placeholder incase needed in the future
        self.N_v = N_v
        self.N_v_err = N_v_err

class RebinnedTOFPointList(list):
    def __init__(self, 
            input_list, 
            v_min,
            v_max,
            v_step,
            label = None):
        super().__init__()

        if isinstance(input_list, RawTOFPointList):
            self.v_step = v_step
            self.length = int( (v_max-v_min)/v_step )
            self.v_min = v_min
            self.v_max = self.v_min + self.length*self.v_step
            self.s = input_list.s
            self.label = label

            for i in range(self.length):
                current_v = self.v_min + i*self.v_step
                current_index_list = []
                for j in range(input_list.length):
                    if input_list[j].v >= current_v and input_list[j].v < current_v + self.v_step:
                        current_index_list.append(input_list[j].index)

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
                    for j in current_index_list:
                        rebin_N.append( input_list[j].N_v )
                        weights.append( 1 / (input_list[j].N_v_err**2) )
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

        elif isinstance(input_list, list) and len(input_list) == 2:
            # assume input_list[0] = v, input_list[1]=N_v
            input_list_v = input_list[0]
            input_list_N_v = input_list[1]

            self.v_step = v_step
            self.length = int( (v_max-v_min)/v_step )
            self.v_min = v_min
            self.v_max = self.v_min + self.length*self.v_step
            self.s = None
            self.label = label

            for i in range(self.length):
                current_v = self.v_min + i*self.v_step
                current_index_list = []
                for j in range(len(input_list_v)):
                    if input_list_v[j] >= current_v and input_list_v[j] < current_v + self.v_step:
                        current_index_list.append(j)

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
                    for j in current_index_list:
                        rebin_N.append( input_list_N_v[j] )
                    rebin_N = np.array(rebin_N)

                    mean_rebin_N = np.mean(rebin_N)
                    mean_rebin_N_err = np.std(mean_rebin_N) / len(rebin_N)

                    self.append(RebinnedTOFPoint(
                        index = i, 
                        s = self.s, 
                        label = self.label,
                        origin_indices_list = current_index_list,
                        v = current_v + 0.5 * self.v_step,
                        N_v = mean_rebin_N,
                        N_v_err = mean_rebin_N_err
                    ))
        else:
            raise TypeError(f"{type(input_list)} is no valid data type for input_list")

    @np.errstate(all='raise')
    def find_peak_velocity(self, 
                     v_low = None, v_high = None,
                     gausstype='fourpoints', 
                     a0 = 1.0, mu0 = 0.0, sigma0 = 1.0, d0 = None, 
                     do_plot=False, do_print=False):

        if v_low is None or v_low < self.v_min:
            v_low = self.v_min
        if v_high is None or v_high > self.v_max:
            v_high = self.v_max

        xdata = []
        ydata = []
        sigmas = []
        for i in self.index:
            if self[i].v >= v_low and self[i].v <= v_high:
                xdata.append(self[i].v)
                ydata.append(self[i].N_v)
                sigmas.append(self[i].N_v_err)

        #xdata = self.v
        #ydata = self.N_v
        #sigmas = self.N_v_err

        if gausstype == 'fourpoints':
            if d0 is None:
                print("Warning: no initial value given for d0, setting d0 = 0.")
                d0 = 0
            parameters, pcov = scipy.optimize.curve_fit(gauss4, xdata, ydata, p0=[a0, mu0, sigma0, d0], sigma=sigmas)
            fit_A, fit_B, fit_C, fit_D = parameters
            err_A, err_B, err_C, err_D = np.sqrt(np.diag(pcov)) # diagonal elements are one standart deviation of parameter
            fit_y = gauss4(xdata, fit_A, fit_B, fit_C, fit_D)
            if do_print == True:
                print(f"Fit parameters: A={fit_A:.2f}+-{err_A:.2f}, mean={fit_B:.4f}+-{err_B:.4f}, sigma={fit_C:.2f}+-{err_C:.2f}, D={fit_D:.2f}+-{err_D:.2f}")
        elif gausstype == 'threepoints':
            if d0 is not None:
                print(f"Warning, set value for d0 = {d0} will not be used as d is not a parameter for gauss3(x, a, x0, sigma).")
            parameters, pcov = scipy.optimize.curve_fit(gauss3, xdata, ydata, p0=[a0, mu0, sigma0])
            fit_A, fit_B, fit_C= parameters
            err_A, err_B, err_C= np.sqrt(np.diag(pcov)) # diagonal elements are one standart deviation of parameter
            fit_y = gauss3(xdata, fit_A, fit_B, fit_C)
            if do_print == True:
                print(f"Fit parameters: A={fit_A:.2f}+-{err_A:.2f}, mean={fit_B:.4f}+-{err_B:.4f}, sigma={fit_C:.2f}+-{err_C:.2f}")
        else:
            print(f"Error: gausstype= {gausstype} not recocnized")
            return 
        
        residuen = (fit_y-ydata)
        try:
            r_s = residuen/sigmas
        except:
            print("Encountered error when dividing by sigma")
            r_s =residuen/fit_y
        chi2 = np.sum(( r_s )**2)
        if (len(xdata)-1) == 0:
            chi2_reduced = np.float('nan')
        else:
            chi2_reduced = chi2/(len(xdata)-1)

        peak_velocity = fit_B
        peak_velocity_err = err_B
        if do_plot == True:
            plt.errorbar(xdata, ydata, sigmas, fmt='.', label=self.label)
            plt.plot(xdata, fit_y, marker='.', label=f"Best Gaussian Fit, Reduced Chi2 = {chi2_reduced:.2E}")
            plt.legend()
            plt.xlabel("Velocity [m/s]")
            plt.ylabel(r"counts [$(m/s)^{-1}$]")
            plt.title("Find peak of rebinned velocity distribution")

        return peak_velocity, peak_velocity_err

    def find_mean_velocity(self, do_print=False, do_newcode=True):
        # search for the mean velocity
        # errors due to poisson statistics are typically zero, because of large numbers
        totalcounts = np.sum(self.N_v)
        v = np.array(self.v)
        v_err = np.array(self.v_err)
        N_v = np.array(self.N_v)
        N_v_err = np.array(self.N_v_err)

        mean_vel = np.sum( N_v * v ) / totalcounts 

        if not do_newcode:
            totalcountserrorsquared = np.sum( N_v_err**2 )
            partialerrorssquared = (N_v_err)**2 * v**2
            mean_vel_err = np.sqrt( np.sum(partialerrorssquared) + totalcountserrorsquared*mean_vel**2 ) / totalcounts # probably mistake in (a - b)**2 != a**2 + b**2

        if do_newcode:
            # error propagation should be correct on the new code
            mean_vel_err = np.sqrt( np.sum( ((v-mean_vel) * N_v_err)**2  +  (N_v*v_err)**2 ) ) / totalcounts

        if do_print == True:
            print(f"Mean velocity for {self.label} = {mean_vel:.4f} +- {mean_vel_err:.4f} m/s")

        return mean_vel, mean_vel_err

    def find_rms_velocity(self, do_print=False):
        # search for the mean velocity
        # errors due to poisson statistics are typically zero, because of large numbers
        totalcounts = np.sum(self.N_v)
        v = np.array(self.v)
        v_err = np.array(self.v_err)
        N_v = np.array(self.N_v)
        N_v_err = np.array(self.N_v_err)

        rms_vel = np.sqrt( np.sum( N_v * v**2 ) / totalcounts )

        rms_vel_err = np.sqrt( np.sum( ( v**2 - rms_vel**2)**2 * N_v_err**2  +  (v * N_v * v_err)**2 ) ) / (totalcounts * rms_vel)

        if do_print == True:
            print(f"Root mean squared velocity for {self.label} = {rms_vel:.4f} +- {rms_vel_err:.4f} m/s")

        return rms_vel, rms_vel_err

    @property
    def index(self):
        a = []
        for i in range(self.length):
            a.append(self[i].index)
        return a

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

# Gaussian functions
def gauss3(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def gauss4(x,a,x0,sigma, d):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + d

