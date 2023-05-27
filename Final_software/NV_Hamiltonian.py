import numpy as np
from NV_Constructure import NV_axis_Constructure
from NV_Utils import *

''' Class for working with Hamitonian of NV center '''


class NV_Hamiltonian(NV_axis_Constructure):
    def __init__(self):
        NV_axis_Constructure.__init__(self)
        self.nv_axs = self.u_axs.copy()
        self.Projection_UnitVec = Projection_Matrix(self.u[:, :, np.newaxis])

        # Addition method
        self.cutoff = 0.60
        self.project = True

    def Projection_B2U(self, B_vec):
        return np.dot(self.nv_axs, B_vec.copy()) if self.project else B_vec.copy()

    def Projection_B2U_V3(self, B_vec):
        u_arr_ori = self.u.copy()
        B_along_vec = np.dot(self.Projection_UnitVec, B_vec)
        B_ortho_vec = B_vec - B_along_vec
        B_ortho_nv = np.linalg.norm(B_ortho_vec, axis=1)
        B_along_nv = u_arr_ori @ B_vec
        return np.hstack((B_ortho_nv, np.zeros((B_along_nv.shape)), B_along_nv)).reshape(4, 3, 1)

    def Find_Eigen_Allaxis(self, B_vec):
        # B_u = self.Projection_B2U(B_vec)
        B_u = self.Projection_B2U_V3(B_vec)
        H_mag = (self.gam * np.dot(self.S.T, B_u)).T.reshape(4, 3, 3)
        H_zero = self.D * (self.Sz.T @ self.Sz)
        eigenval, eigenvec = np.linalg.eig(H_zero + H_mag)
        return eigenval, eigenvec

    def A_Frequecies_B(self, B_vec, get_zero=False):
        eigenVal, eigenVec = self.Find_Eigen_Allaxis(B_vec)
        cutoff_eigVec = np.zeros((eigenVec.shape))
        cutoff_eigVec[np.where(eigenVec >= self.cutoff)] = 1
        max_idx = np.argmax(cutoff_eigVec, 2)
        Frequency = np.take_along_axis(eigenVal, max_idx, axis=1)
        if get_zero:
            return Frequency[:, 0], Frequency[:, 1], Frequency[:, 2]
        return Frequency[:, 0], Frequency[:, 2]

    def A_Transition_B(self, B_vec):
        MsPlus, MsZero, MsMinus = self.A_Frequecies_B(
            B_vec=B_vec, get_zero=True)
        return MsPlus - MsZero, MsMinus - MsZero

    def _Plot_show(self):
        self.Plot_axis()
