import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider, Button, CheckButtons


def rotate_mat(theta, phi):
    return np.array([[np.cos(theta) * np.cos(phi), np.sin(phi), -np.sin(theta) * np.cos(phi)], 
                     [-np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 
                     [np.sin(theta), 0, np.cos(theta)]]).T


class NVstructure:
    def __init__(self):
        # Original NV axis
        u1_ori = np.array([[1], [1], [1]]) / 3 ** (1 / 2)
        u2_ori = np.array([[1], [-1], [-1]]) / 3 ** (1 / 2)
        u3_ori= np.array([[-1], [-1], [1]]) / 3 ** (1 / 2)
        u4_ori = np.array([[-1], [1], [-1]]) / 3 ** (1 / 2)
        self.u_arr = np.array((u1_ori, u2_ori, u3_ori, u4_ori)).reshape(4, 3).T

        # Magnetic field vector
        self.B = 1
        self.bTheta, self.bPhi = 0, 0
        self.B_vec = self.B * np.array([[np.sin(self.bTheta) * np.cos(self.bPhi)], [np.sin(self.bTheta) * np.sin(self.bPhi)], [np.cos(self.bTheta)]])

        # Addition plot
        self.color_lst = ['maroon', 'orange', 'royalblue', 'lime', 'cyan', 'yellow', 'darkviolet', 'hotpink']
        self.on_B = False
        self.rTheta, self.rPhi = 0, 0


    def Plot_Structure(self):
        fig = plt.figure('NV Structure')
        ax = plt.axes(projection='3d')
        for i in range(self.u_arr.shape[1]):
            ax.plot3D([0, self.u_arr[0, i]], [0, self.u_arr[1, i]], [0, self.u_arr[2, i]], lw=3, color=self.color_lst[i], alpha=0.5, label=f'NV #{i+1}')
        ax.quiver([0], [0], [0], self.B_vec[0, 0], self.B_vec[1, 0], self.B_vec[2, 0], color='black', label='B')
        plt.subplots_adjust(left=0.155, bottom=0.3, top=0.912, right=0.77)
        plt.title('NV axis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend(bbox_to_anchor=(1.25, 0.5), loc="center left")

        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'
        axB = plt.axes([0.25, 0.2, 0.65, 0.02], facecolor=axcolor)
        axTheta = plt.axes([0.25, 0.15, 0.65, 0.02], facecolor=axcolor)
        axPhi = plt.axes([0.25, 0.1, 0.65, 0.02], facecolor=axcolor)

        sB = Slider(axB, 'B (mT)', 0, 2, valinit=1)
        sTheta = Slider(axTheta, 'Theta (degree)', -180, 180, valinit=0)
        sPhi = Slider(axPhi, 'Phi (degree)', -180, 180, valinit=0)

        def update_NVRotate(val):
            ax.cla()
            theta = sTheta.val * np.pi / 180
            phi = sPhi.val * np.pi / 180
            B = sB.val
            if self.on_B:
                self.bTheta, self.bPhi = theta, phi
                self.B = B
                self.B_vec = self.B * np.array([[np.sin(theta) * np.cos(phi)], [np.sin(theta) * np.sin(phi)], [np.cos(theta)]])
                ax.quiver([0], [0], [0], self.B_vec[0, 0], self.B_vec[1, 0], self.B_vec[2, 0], color='black', label='B')
                rot_mat = rotate_mat(self.rTheta, self.rPhi)
                self.u_arr_work = rot_mat @ self.u_arr.copy()
                for i in range(self.u_arr.shape[1]):
                    ax.plot3D([0, self.u_arr_work[0, i]], [0, self.u_arr_work[1, i]], [0, self.u_arr_work[2, i]], lw=3, color=self.color_lst[i], alpha=0.5, label=f'NV #{i+1}')
            else:
                self.rTheta, self.rPhi = theta, phi
                self.B_vec = self.B * np.array([[np.sin(self.bTheta) * np.cos(self.bPhi)], [np.sin(self.bTheta) * np.sin(self.bPhi)], [np.cos(self.bTheta)]])
                ax.quiver([0], [0], [0], self.B_vec[0, 0], self.B_vec[1, 0], self.B_vec[2, 0], color='black', label='B')
                rot_mat = rotate_mat(theta, phi)
                self.u_arr_work = rot_mat @ self.u_arr.copy()
                for i in range(self.u_arr.shape[1]):
                    ax.plot3D([0, self.u_arr_work[0, i]], [0, self.u_arr_work[1, i]], [0, self.u_arr_work[2, i]], lw=3, color=self.color_lst[i], alpha=0.5, label=f'NV #{i+1}')
            ax.set_title('NV axis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend(bbox_to_anchor=(1.25, 0.5), loc="center left")
            fig.canvas.draw()

        sB.on_changed(update_NVRotate)
        sTheta.on_changed(update_NVRotate)
        sPhi.on_changed(update_NVRotate)

        def on_off(event):
            print(check.get_status()[0] == True)
            self.on_B = check.get_status()[0]
        rax = plt.axes([0.05, 0.4, 0.1, 0.1])
        check = CheckButtons(rax, ['B'])
        check.on_clicked(on_off)

        def reset(event):
            sTheta.reset()
            sPhi.reset()
            sB.reset()
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        button.on_clicked(reset)

        plt.show()


if __name__ == '__main__':
    NV_str = NVstructure()
    NV_str.Plot_Structure()

