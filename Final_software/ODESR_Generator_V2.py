import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.io import savemat

from NV_Hamiltonian import NV_Hamiltonian

''' 
------- version 2 -------
 - Defining the amp and wid of flip gaussian by fitting the real OD-ESR spectra
'''


class GEN_ODESR(NV_Hamiltonian):
    def __init__(self):
        NV_Hamiltonian.__init__(self)

        ''' Parameters for Generator '''
        # For OD-ESR plot
        self.fre_init = 1.2  # GHz
        self.fre_end = 4.5  # GHz
        self.num_fre = int(5e3)
        self.fre_arr = np.linspace(self.fre_init, self.fre_end, self.num_fre)
        self.amp = 2.5e-4
        self.wid = 1e-2
        # self.amp = 5.1749e-05 # real
        # self.wid = 0.00350958 # real

        # For magnetic field
        self.Bval = 0
        self.theta = 0 * np.pi / 180
        self.phi = 0 * np.pi / 180

        ''' Program results '''
        self.TransPlus = None
        self.TransMinus = None
        self.NV_axis = {}
        self.All_PLdata = [[]]*8
        PL_dict = None
        self.ODESR_label = None

        ''' Addition plot '''
        self.color_lst = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf",
                          "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"]
        self.B_min, self.B_max = 0.1, 50.0
        self.theta_min, self.theta_max = 0.0, 180.0
        self.phi_min, self.phi_max = -180.0, 180.0

    @staticmethod
    def flip_gaussian(x, amp, cen, wid):
        return 1-(amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

    def Mark_NV(self):
        trans = np.append(self.TransPlus.copy(), self.TransMinus.copy())
        trans_sort_half = np.sort(trans)[:4]
        NV_number = np.array([1,2,3,4]*2)
        a = []
        for fre_now in trans_sort_half:
            close_idx = np.where(fre_now == trans)[0][0]
            a.append(NV_number[close_idx])
            trans = np.delete(trans, close_idx)
            NV_number = np.delete(NV_number, close_idx)
        return np.array(a)

    def Generate_ODESR(self, plus, minus):
        plus_PL = np.array(
            [abs(self.flip_gaussian(self.fre_arr, self.amp, cen, self.wid)) for cen in plus])
        minus_PL = np.array(
            [abs(self.flip_gaussian(self.fre_arr, self.amp, cen, self.wid)) for cen in minus])
        return plus_PL.T, minus_PL.T

    def Plot_ODESR(self):
        fig = plt.figure('ODESR')
        ax = plt.axes()

        # First Calculated Part
        B_vec = self.Bval * np.array([[np.sin(self.theta) * np.cos(self.phi)], [
                                     np.sin(self.theta) * np.sin(self.phi)], [np.cos(self.theta)]])
        self.TransPlus, self.TransMinus = self.A_Transition_B(B_vec=B_vec)
        # self.TransPlus, self.TransMinus = self.A_Frequecies_B(B_vec=B_vec)
        self.ODESR_label = self.Mark_NV()
        PL_TransPlus, PL_TransMinus = self.Generate_ODESR(
            self.TransPlus, self.TransMinus)

        # Plot Part
        save_num = 0
        for idx in range(4):  # 4 difference NV-axis
            PL_minus, PL_plus = PL_TransMinus[:, idx], PL_TransPlus[:, idx]
            line_minus, = plt.plot(
                self.fre_arr, PL_minus, '--', label=f'NV {idx + 1}' + '_-', color=self.color_lst[idx])
            line_plus, = plt.plot(
                self.fre_arr, PL_plus, '-', label=f'NV {idx + 1}' + '_+', color=self.color_lst[idx])
            self.NV_axis[f'{idx + 1}' + '_-'] = line_minus
            self.NV_axis[f'{idx + 1}' + '_+'] = line_plus
            self.All_PLdata[save_num] = PL_minus
            self.All_PLdata[save_num+1] = PL_plus
            save_num += 2
        plt.subplots_adjust(left=0.15, bottom=0.355, top=0.906, right=0.799)
        plt.title('OD-ESR spectra')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('PL (a.u.)')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

        # Widgets on Plot Window
        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'
        axB = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
        axTheta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axPhi = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

        self.sB = Slider(axB, 'B (mT)', self.B_min,
                         self.B_max, valinit=self.Bval)
        self.sTheta = Slider(axTheta, 'Theta (degree)', self.theta_min,
                             self.theta_max, valinit=self.theta, valstep=0.1)
        self.sPhi = Slider(axPhi, 'Phi (degree)', self.phi_min,
                           self.phi_max, valinit=self.phi, valstep=0.1)

        # Update Plot Part
        def update(val):

            B = self.sB.val * 0.001
            theta = self.sTheta.val * np.pi / 180
            phi = self.sPhi.val * np.pi / 180

            # Calculated Part
            B_vec = B * np.array([[np.sin(theta) * np.cos(phi)],
                                  [np.sin(theta) * np.sin(phi)], [np.cos(theta)]])
            self.TransPlus, self.TransMinus = self.A_Transition_B(B_vec=B_vec)
            # self.TransPlus, self.TransMinus = self.A_Frequecies_B(B_vec=B_vec)
            self.ODESR_label = self.Mark_NV()
            PL_TransPlus, PL_TransMinus = self.Generate_ODESR(
                self.TransPlus, self.TransMinus)
            save_num = 0

            # Plot Part
            for idx in range(4):
                PL_minus, PL_plus = PL_TransMinus[:, idx], PL_TransPlus[:, idx]
                self.NV_axis[f'{idx + 1}' + '_-'].set_ydata(PL_minus)
                self.NV_axis[f'{idx + 1}' + '_+'].set_ydata(PL_plus)
                self.All_PLdata[save_num] = PL_minus
                self.All_PLdata[save_num+1] = PL_plus
                save_num += 2
            fig.canvas.draw_idle()

        self.sB.on_changed(update)
        self.sTheta.on_changed(update)
        self.sPhi.on_changed(update)

        # The Method for Reset Widget
        def reset(event):
            self.sB.reset()
            self.sTheta.reset()
            self.sPhi.reset()
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')
        button.on_clicked(reset)

        # The Method for Save Widget
        def save(event):
            print('Save Now')
            PL_init = 0
            for PL_i in self.All_PLdata:
                PL_init += PL_i
            PL_init -= 7
            PL_dict = {'pl': PL_init, 'xvals': self.fre_arr*1e9,
                       'ref': np.ones(self.num_fre), 'sig': PL_init, 'ODESR_label': self.ODESR_label}
            B = np.round(self.sB.val * 0.001, 3)
            theta = np.round(self.sTheta.val, 3)
            phi = np.round(self.sPhi.val, 3)
            try:
                savemat(
                    f'./Mat_file/B_{B}_Theta_{theta}_Phi_{phi}.mat', PL_dict)
                print('Save Sucessfully')
            except Exception:
                print('Save Unsucessfully')
        save_ax = plt.axes([0.65, 0.025, 0.1, 0.04])
        save_button = Button(
            save_ax, 'Save', color=axcolor, hovercolor='0.975')
        save_button.on_clicked(save)

        def preview(event):
            figPreview = plt.figure('Preview Window')
            figPreview.clear()
            PL_init = 0
            for PL_i in self.All_PLdata:
                PL_init += PL_i
            PL_init -= 7
            B = np.round(self.sB.val * 0.001, 3)
            theta = np.round(self.sTheta.val, 2)
            phi = np.round(self.sPhi.val, 2)
            plt.plot(self.fre_arr, PL_init, 'r-')
            plt.title(f'B : {B}, Theta : {theta}, Phi : {phi}')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('PL (a.u.)')
            plt.show()
        preview_ax = plt.axes([0.48, 0.025, 0.12, 0.04])
        preview_button = Button(preview_ax, 'PL Preview',
                                color=axcolor, hovercolor='0.975')
        preview_button.on_clicked(preview)

        def plotStructure(event):
            B = self.sB.val * 0.001
            theta = self.sTheta.val * np.pi / 180
            phi = self.sPhi.val * np.pi / 180

            # Calculated Part
            B_vec = B * np.array([[np.sin(theta) * np.cos(phi)],
                                  [np.sin(theta) * np.sin(phi)], [np.cos(theta)]])
            B_vec = B_vec * 1000 / self.B_max
            self.Plot_axis(B_vec=B_vec)
        plotNV_ax = plt.axes([0.31, 0.025, 0.12, 0.04])
        plot3D_button = Button(plotNV_ax, '3D strcture',
                               color=axcolor, hovercolor='0.975')
        plot3D_button.on_clicked(plotStructure)

        plt.show()

    def Plot_TransitionWithB(self):
        fig = plt.figure('ODESR')
        ax = plt.axes()

        # First Calculated Part
        B_vec = self.Bval * np.array([[np.sin(self.theta) * np.cos(self.phi)], [
                                     np.sin(self.theta) * np.sin(self.phi)], [np.cos(self.theta)]])
        pass

    def Calculate(self):
        self.Plot_ODESR()


if __name__ == '__main__':
    Gen = GEN_ODESR()
    Gen.Calculate()
