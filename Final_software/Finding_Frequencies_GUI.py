import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from scipy.spatial.distance import cdist
from sklearn import cluster
import numpy as np
import pandas as pd

from CallMat_as_Me_Utils import All_MatData
from Initial_Finding_Frequencies import Initial_Frequencies
from Fit_Finding_Frequencies import Fitted_Frequencies


class Find_Fre_GUI:
    def __init__(self):
        # Parents
        self.inital_fre = Initial_Frequencies()
        self.fitted_fre = Fitted_Frequencies()

        # Class attributes
        self.mat_path = None
        self.fre, self.pl = None, None

        # Class parameter
        self.cuton_val = 1
        self.dogmatic_On = False
        self.number_of_NV = 8
        self.kmin, self.kmax =  1, 11

        # Method output 
        self.initial_frequencies = None
        self.distortions = np.array([])
        self.inertias = np.array([])
        self.fitting_error = np.array([])
        self.elbow_cutval = 0

    @staticmethod
    def Normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    def Get_InitFre(self):
        if self.dogmatic_On:
            self.initial_frequencies = self.inital_fre.Run_Dogmatic(self.fre, self.pl, self.cuton_val, self.number_of_NV)
        else:
            self.initial_frequencies = self.inital_fre.Run_Simple(self.fre, self.pl, self.cuton_val)

    def Plot_GUI(self):
        fig = plt.figure('Finding Frequencies via OD-ESR spectra')
        ax = plt.axes()

        # Plot part
        plt.plot(self.fre / 1e9, self.pl, 'r.-')
        cut_line, = plt.plot(self.fre / 1e9, 0 * self.fre / 1e9 + 1, '-', color='black')

        plt.subplots_adjust(left=0.138, bottom=0.28, top=0.902, right=0.842)
        plt.title('OD-ESR spectra')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('PL (a.u.)')

        # Widget part

        axamp = plt.axes([0.9, 0.27, 0.0225, 0.63])
        amp_slider = Slider(
            ax=axamp,
            label="Cut-On",
            valmin=self.pl.min(),
            valmax=self.pl.max(),
            valinit=1,
            orientation="vertical")

        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')

        initial_ax = plt.axes([0.6, 0.025, 0.1, 0.04])
        initial_button = Button(initial_ax, 'Initial', hovercolor='0.975')

        fitted_ax = plt.axes([0.4, 0.025, 0.1, 0.04])
        fitted_button = Button(fitted_ax, 'Fitted', hovercolor='0.975')

        text_ax = plt.axes([0.2, 0.025, 0.1, 0.04])
        text_box = TextBox(text_ax, 'Number of NV  ')

        check_ax = plt.axes([0.05, 0.1, 0.2, 0.1])
        check_box = CheckButtons(check_ax, ['Dogmatic'])

        approx_ax = plt.axes([0.8, 0.1, 0.1, 0.04])
        approx_button = Button(approx_ax, 'Approx K', hovercolor='0.975')

        def update(val):
            cut_line.set_ydata(0 * self.fre + val)
            self.cuton_val = val
            fig.canvas.draw_idle()
        amp_slider.on_changed(update)

        def check_dogmatic(val):
            self.dogmatic_On = check_box.get_status()[0]
        check_box.on_clicked(check_dogmatic)

        def get_numberNV(val):
            self.number_of_NV = int(val)
        text_box.on_submit(get_numberNV)

        def inital_frequencies_plot(val):
            self.Get_InitFre()
            InitPreview = plt.figure('Preview Initial Frequencies')
            InitPreview.clear()
            for fre_now in self.initial_frequencies:
                plt.vlines(fre_now, self.pl.min(), self.pl.max(), label=f'{round(fre_now / 1e9, 2)} GHz')
            plt.plot(self.fre, self.pl, 'r.-')
            plt.subplots_adjust(left=0.13, bottom=0.22, top=0.88, right=0.792)
            plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('PL (a.u.)')

            def save_init(val):
                df = pd.DataFrame(np.vstack((np.sort(self.initial_frequencies), np.append(self.ODESR_label, self.ODESR_label[::-1]))).T, columns=['Initial_frequncies (Hz)', 'NV_label'])
                df = df.astype({'Initial_frequncies (Hz)':np.float32, 'NV_label':int})
                try:
                    df.to_csv(f'/Frequencies_CSV_file/init_fre_{self.working_name[0]}.csv', index=False)
                    print('Save Succuessfully')
                except Exception:
                    print('Save Unsuccessfully')
            save_init_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
            save_init_button = Button(save_init_ax, 'Save', hovercolor='0.975')
            save_init_button.on_clicked(save_init)

            plt.show()
        initial_button.on_clicked(inital_frequencies_plot)

        def fitted_frequncies_plot(val):
            self.Get_InitFre()
            self.fitted_frequencies, fitted_model = self.fitted_fre.Run_Fitting(self.fre, self.pl, self.initial_frequencies, get_model=True)
            FitPreview = plt.figure('Preview Fitted Frequencies')
            FitPreview.clear()
            for idx, fre_now in enumerate(self.fitted_frequencies):
                plt.vlines(fre_now, fitted_model[idx].min(), fitted_model[idx].max(), label=f'{round(fre_now / 1e9, 2)} GHz')
                plt.plot(self.fre, fitted_model[idx], '.-')
            plt.subplots_adjust(left=0.13, bottom=0.22, top=0.88, right=0.792)
            plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('PL (a.u.)')

            def save_fit(val):
                df = pd.DataFrame(np.vstack((np.sort(self.fitted_frequencies), np.append(self.ODESR_label, self.ODESR_label[::-1]))).T, columns=['Fitted_frequncies (Hz)', 'NV_label'])
                df = df.astype({'Fitted_frequncies (Hz)':np.float32, 'NV_label':int})
                try:
                    df.to_csv(f'./Frequencies_CSV_file/fit_fre_{self.working_name[0]}.csv', index=False)
                    print('Save Succuessfully')
                except Exception:
                    print('Save Unsuccessfully')
            save_fit_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
            save_fit_button = Button(save_fit_ax, 'Save', hovercolor='0.975')
            save_fit_button.on_clicked(save_fit)

            plt.show()
        fitted_button.on_clicked(fitted_frequncies_plot)

        def approximation_k(event):
            # Check the input
            if len(self.fitting_error) == 0 or self.elbow_cutval != self.cuton_val:
                self.elbow_cutval = self.cuton_val
                work_x = self.fre.copy()
                dim = len(work_x.shape)
                if dim == 1:
                    work_x = np.vstack((np.arange(work_x.shape[0]), work_x)).T
                else:
                    work_x = work_x.T
                self.distortions = np.array([])
                self.inertias = np.array([])
                self.fitting_error = np.array([])
                cutoff_x = work_x[np.where(self.pl <= self.cuton_val)]
                for k in range(self.kmin, self.kmax):
                    kmeanModel = cluster.KMeans(n_clusters=k)
                    kmeanModel.fit(cutoff_x)
                    center = kmeanModel.cluster_centers_
                    fitting_model = self.fitted_fre.Get_BestFit(self.fre, self.pl, center[:, 1])
                    self.fitting_error = np.append(self.fitting_error, np.mean((self.pl.copy() - fitting_model)**2))
                    self.distortions = np.append(self.distortions, sum(np.min(cdist(cutoff_x, center, 'euclidean'), axis=1)) / cutoff_x.shape[0])
                    self.inertias = np.append(self.inertias, kmeanModel.inertia_)
            norm_dis, norm_iner, norm_fit_err = self.Normalize(self.distortions), self.Normalize(self.inertias), self.Normalize(self.fitting_error)

            # Plot the elbow plot
            ElbowPreview = plt.figure('Preview Fitted Frequencies')
            ElbowPreview.clear()
            k_arr = np.arange(self.kmin, self.kmax)
            # plt.plot(k_arr, norm_iner, '-*', label = 'Inertias', lw=3, s=10)
            plt.plot(k_arr, norm_dis, '-d', label = 'Distortion', lw=3)
            plt.plot(k_arr, norm_fit_err, '.-', label = 'Fit Error', lw=3)
            plt.legend()
            plt.grid()
            plt.xlabel('Number of K')
            plt.ylabel('Normalized error (a.u.)')

            plt.show()
        approx_button.on_clicked(approximation_k)

        def reset(event):
            amp_slider.reset()
        reset_button.on_clicked(reset)

        plt.show()

    @staticmethod
    def Check_KeyboardInput(key_input):
        try:
            key_input = int(key_input)
        except ValueError:
            key_input = str(key_input)
        return key_input

    def Calculate(self, mat_path):
        while True:
            self.mat_path = mat_path
            Mat_Files = All_MatData(self.mat_path).CallMatData()
            key_lst = list(Mat_Files.keys())
            for idx, key in enumerate(key_lst):
                print(f'[{idx+1}] {key}')
            print('[r] Restart')
            print('[q] Quit')
            
            keyboard_input = self.Check_KeyboardInput(input('Please Enter: ').lower())
            if keyboard_input == 'q':
                print('Quit')
                break
            elif keyboard_input == 'r':
                continue
            elif keyboard_input in np.arange(len(key_lst) + 1):
                working_mat = Mat_Files[key_lst[keyboard_input-1]]
                self.working_name = key_lst[keyboard_input-1].split('.mat')
                print(f'Number: {keyboard_input} Name: {key_lst[keyboard_input-1]}')

                self.fre, self.pl = working_mat.xvals, working_mat.pl
                self.ODESR_label = working_mat.ODESR_label
                self.dogmatic_On = False
                self.Plot_GUI()
                
            else:
                print('[-] Can not find your input number or command, please enter again')

if __name__ == '__main__':

    mat_path = './Mat_file/'
    Find_Fre_GUI().Calculate(mat_path=mat_path)        
    