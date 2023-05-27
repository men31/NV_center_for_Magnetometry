import numpy as np
import lmfit

class Fitted_Frequencies:
    def __init__(self):
        # Inital parameter for inital guess
        self.init_amp = 0
        self.init_wid = 0

        # Constraint parameter for fitting
        self.number_of_NV = 8
        self.val_amp, self.val_wid = 0.05, 0.01
        self.min_amp, self.min_cen, self.min_wid = 0, 1, 0
        self.max_amp, self.max_cen, self.max_wid = 0.1, 5, 0.05

        # Method outputs
        self.results = None
        self.opt_dict = None
        self.fitted_model = None

    '''Statistical model to fit the OD-ESR spectra'''
    @staticmethod
    def flip_gaussian(x, amp, cen, wid):
        return 1-(amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

    @staticmethod
    def minus_gaussian(x, amp, cen, wid):
        return - (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

    def Make_Model_Params(self, init_center): # making parameters for fitting
        for i in range(self.number_of_NV):
            if i == 0:
                mod = lmfit.Model(self.flip_gaussian, prefix=f'fg{i}_')
                mod.set_param_hint(f'fg{i}_amp', value=self.val_amp, min=self.min_amp, max=self.max_amp)
                mod.set_param_hint(f'fg{i}_cen', value=init_center[i] / 1e9, min=self.min_cen, max=self.max_cen)
                mod.set_param_hint(f'fg{i}_wid', value=self.val_wid, min=self.min_wid, max=self.max_wid)
            else:
                mod += lmfit.Model(self.minus_gaussian, prefix=f'fg{i}_')
                mod.set_param_hint(f'fg{i}_amp', value=self.val_amp, min=self.min_amp, max=self.max_amp)
                mod.set_param_hint(f'fg{i}_cen', value=init_center[i] / 1e9, min=self.min_cen, max=self.max_cen)
                mod.set_param_hint(f'fg{i}_wid', value=self.val_wid, min=self.min_wid, max=self.max_wid)
        pars = mod.make_params()
        return mod, pars
    
    def Get_BestFit(self, fre, pl, init_center):
        for i in range(len(init_center)):
            if i == 0:
                mod = lmfit.Model(self.flip_gaussian, prefix=f'fg{i}_')
                mod.set_param_hint(f'fg{i}_amp', value=self.val_amp, min=self.min_amp, max=self.max_amp)
                mod.set_param_hint(f'fg{i}_cen', value=init_center[i] / 1e9, min=self.min_cen, max=self.max_cen)
                mod.set_param_hint(f'fg{i}_wid', value=self.val_wid, min=self.min_wid, max=self.max_wid)
            else:
                mod += lmfit.Model(self.minus_gaussian, prefix=f'fg{i}_')
                mod.set_param_hint(f'fg{i}_amp', value=self.val_amp, min=self.min_amp, max=self.max_amp)
                mod.set_param_hint(f'fg{i}_cen', value=init_center[i] / 1e9, min=self.min_cen, max=self.max_cen)
                mod.set_param_hint(f'fg{i}_wid', value=self.val_wid, min=self.min_wid, max=self.max_wid)
        pars = mod.make_params()
        self.results = mod.fit(pl, pars, x=fre/1e9)
        fit_pl = self.results.best_fit
        return fit_pl

    def Run_Fitting(self, fre, pl, init_center, get_model=False):
        mod, pars = self.Make_Model_Params(init_center)
        self.results = mod.fit(pl, pars, x=fre/1e9)

        # Output
        comps = self.results.eval_components()
        self.opt_dict = self.results.best_values
        self.fitted_model = [comps[f'fg{i}_'] if i == 0 else comps[f'fg{i}_'] + 1 for i in range(self.number_of_NV)]
        self.fitted_frequencies = np.sort([val for key, val in self.opt_dict.items() if 'cen' in key]) * 1e9
        if get_model:
            return self.fitted_frequencies, self.fitted_model
        return self.fitted_frequencies
