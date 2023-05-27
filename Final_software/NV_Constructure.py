import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from NV_Utils import *



''' Class for construction the NV structure'''
class NV_axis_Constructure:
    def __init__(self):
        # Reference Axis
        x_axs = np.array([[1], [0], [0]])
        y_axs = np.array([[0], [1], [0]])
        z_axs = np.array([[0], [0], [1]])
        self.n_axs = np.array((x_axs, y_axs, z_axs)).reshape(3, 3)

        # Original NV axis
        u1_ori = np.array([[1], [1], [1]]) / 3 ** (1 / 2)
        u2_ori = np.array([[1], [-1], [-1]]) / 3 ** (1 / 2)
        u3_ori= np.array([[-1], [-1], [1]]) / 3 ** (1 / 2)
        u4_ori = np.array([[-1], [1], [-1]]) / 3 ** (1 / 2)
        u_arr_ori = np.array((u1_ori, u2_ori, u3_ori, u4_ori)).reshape(4, 3)

        # Rotated NV axis
        self.rot_theta = 54.735610315 * np.pi / 180
        self.rot_phi = 45 * np.pi / 180
        rotating_matrix = Rotate_Sphere(self.rot_theta, self.rot_phi)
        self.rotate_bool = False
        u_rot = (rotating_matrix.T @ u_arr_ori.T).T
        if self.rotate_bool:
            self.u = np.array((u_rot[0], u_rot[1], u_rot[2], u_rot[3]))
        else:
            self.u = u_arr_ori
        self.u_arr = np.array((self.u[0].reshape(3, 1), self.u[1].reshape(3, 1), self.u[2].reshape(3, 1), self.u[3].reshape(3, 1))).reshape(4, 3).T

        # Rotated Reference Axis allowing NV axis
        self._NV_AlignmentType = 0
        self.u_axs = self.SelectCoordinates()
        self.u_dict = {f'NV_{i}': self.u_axs[i] for i in range(len(self.u_axs))}

        # Spin operator
        self.Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        self.Sy = np.array([[0, -1.j, 0], [1.j, 0, -1.j], [0, 1.j, 0]]) / np.sqrt(2)
        self.Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]) 
        self.S = np.array((self.Sx, self.Sy, self.Sz))

        # NV properties
        self.gam = 28.02  # gamma NV = 28.02 GHz / T
        self.E = 5 / 1000  # E = 5 MHz
        self.D = 2.87  # D = 2.87 GHz
        
        self.color_lst = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c","#dede00"]
        
    ''' Method for constructing NV center part'''

    @property
    def NV_AlignmentType(self):
        print("Get value")
        return self._NV_AlignmentType
    
    @NV_AlignmentType.setter
    def NV_AlignmentType(self, CoordinateType):
        print('Set value')
        self._NV_AlignmentType = CoordinateType
        self.nv_axs = self.SelectCoordinates(NV_CoordinateType = CoordinateType)
    
    def SelectCoordinates(self, NV_CoordinateType = 0):
        if NV_CoordinateType in (0, 'default', '0'):
            return np.array([MakeCoordinate_u(u_i) for u_i in self.u])
        elif NV_CoordinateType in (1, '1'):
            u_work = self.u[:, :, np.newaxis]
            All_Orthogonal_Axs = GenerateOrthoAxis(u_work, 2)
            return MakeCoordinate_u_V2(All_Orthogonal_Axs)
        elif NV_CoordinateType in (2, '2'):
            u_work = self.u[:, :, np.newaxis]
            All_Orthogonal_Axs = GenerateOrthoAxis(u_work, 1)
            All_Orthogonal_Axs.insert(1, np.zeros(All_Orthogonal_Axs[0].shape))
            return MakeCoordinate_u_V2(All_Orthogonal_Axs)
        else:
            CoordinateType = int(input('Please enter [0/1/2] : '))
            self.SelectCoordinates(NV_CoordinateType=CoordinateType)
    
    ''' Method for representing NV center formation in 3-dimension'''
    def Plot_axis(self, B_vec=None):
        fig = plt.figure('NV Structure')
        ax = plt.axes(projection='3d')
        for i in range(self.u_arr.shape[1]):
            ax.plot3D([0, self.u_arr[0, i]], [0, self.u_arr[1, i]], [0, self.u_arr[2, i]], lw=3, color=self.color_lst[i], alpha=0.5, label=f'NV #{i+1}')
        if B_vec is not None:
            ax.quiver([0], [0], [0], B_vec[0, 0], B_vec[1, 0], B_vec[2, 0], color='black', label='B')
        plt.subplots_adjust(left=0.155, bottom=0.3, top=0.912, right=0.65)
        plt.title('NV axis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend(bbox_to_anchor=(1.2,0.5), loc="center left")
        plt.show()



