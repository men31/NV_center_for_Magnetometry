import numpy as np
import matplotlib.pyplot as plt

from NV_Hamiltonian import NV_Hamiltonian
from Sorting_Utils import *


class Sorting_RawFre(NV_Hamiltonian):

    def __init__(self): 
        NV_Hamiltonian.__init__(self)

        # class attributes
        self.raw_p, self.raw_m = None, None
        self.u_arr_ori = self.u

        # method attributes
        self.theta_lst = np.linspace(0, np.pi, 361)
        self.phi_lst = np.linspace(-np.pi, np.pi, 721)
        self.omega_lst = np.linspace(-np.pi, np.pi, 721)
        self.canal_width = 1
        self.packages = False

        # method ouput
        self.twice_landscape = None
        self.angle_prob = None
        self.B_val = None
        self.Frequenices_prob = []

    ''' Method for Finding Alpha Angle between The Four NV-axis and Magnetic Vector'''

    def Find_B(self, v1, v2):
        return np.sqrt((v1**2 + v2**2 - v1*v2 - self.D**2)/3) / (self.gam)

    def Find_alpha(self, v1, v2):
        x_val = np.sqrt(((2*v1 - v2 - self.D)*(v1 - 2*v2 + self.D)*(v1 + v2 + self.D))/(9*self.D * (v1**2 + v2**2 - v1*v2 - self.D**2)))
        return np.arccos(x_val), np.arccos(-x_val)

    def Get_allAlpha_BVec(self, plus=True):
        All_alpha = []
        if plus:
            for i in range(len(self.raw_p)):
                alpha_p, alpha_m = self.Find_alpha(self.raw_m[i], self.raw_p[i])
                All_alpha.append(alpha_p)
        else:
            for i in range(len(self.raw_p)):
                alpha_p, alpha_m = self.Find_alpha(self.raw_m[i], self.raw_p[i])
                All_alpha.append(alpha_m)
        return np.array(All_alpha) 

    ''' Method for Making the Search Landscape '''

    def Digging_Lanscape(self, All_alpha):
        landscape = np.zeros((self.theta_lst.shape[0], self.phi_lst.shape[0]))
        init_B = np.real(Make_B_Probability(All_alpha, self.u_arr_ori).astype(np.complex64))
        for idx, u in enumerate(self.u_arr_ori):
            landscape_u = np.zeros((self.theta_lst.shape[0], self.phi_lst.shape[0]))
            for omega in self.omega_lst:
                B_prob_vec = RotateVecAround(self.u_arr_ori[idx][:, np.newaxis], init_B[idx], omega)
                phi_now = np.sum(Find_phi(B_prob_vec, np.array([[1], [0], [0]]), np.array([[0], [1], [0]])))
                theta_now = np.sum(Find_theta(B_prob_vec, np.array([[0], [0], [1]])))
                idx_min_theta = np.argmin((self.theta_lst.copy() - theta_now)**2)
                idx_min_phi = np.argmin((self.phi_lst.copy() - phi_now)**2)
                on_scape = landscape_u[idx_min_theta-self.canal_width:idx_min_theta+1+self.canal_width, 
                                        idx_min_phi-self.canal_width:idx_min_phi+1+self.canal_width]
                on_scape[np.where(on_scape == 0)] -= 5e2
            landscape += landscape_u 
        return landscape

    def Digging__Twice_Lanscape(self):
        All_alpha = self.Get_allAlpha_BVec(plus=True)
        self.landscape_true = self.Digging_Lanscape(All_alpha)
        All_alpha = self.Get_allAlpha_BVec(plus=False)
        self.landscape_false = self.Digging_Lanscape(All_alpha)
        if self.packages:
            return self.landscape_false + self.landscape_true, self.theta_lst, self.phi_lst
        return self.landscape_false + self.landscape_true

    ''' Method for Implementing with the Result'''

    @staticmethod
    def Check_theta_phi(theta_result, phi_result):
        theta_prob = np.pi - theta_result
        if phi_result < 0:
            phi_prob = phi_result + np.pi
        else:
            phi_prob = phi_result - np.pi
        return np.array([[theta_result, phi_result], [theta_prob, phi_prob]])

    @staticmethod
    def SortThePrime(p, m, p_prime, m_prime):
        Trans_work = np.append(m_prime.copy(), p_prime.copy())
        Sorted_Trans_prime = []
        for fre in np.append(m.copy(), p.copy()):
            close_idx = np.argmin((fre - Trans_work)**2)
            Sorted_Trans_prime.append(Trans_work[close_idx])
            Trans_work = np.delete(Trans_work, close_idx)
        m_prime_new, p_prime_new = Sorted_Trans_prime[:4], Sorted_Trans_prime[4:]
        return [p_prime_new, m_prime_new]

    def Find_Minimum_Landscape(self, landscape):
        idx_min_2d = np.unravel_index(landscape.argmin(), landscape.shape)
        angle_prob = self.Check_theta_phi(self.theta_lst[idx_min_2d[0]], self.phi_lst[idx_min_2d[1]])
        return angle_prob # theta, phi 
    
    def Plot_TwiceLandscape(self, mode='both'):
        if mode in ['plus', 'true', 1]:
            plt.contourf(self.phi_lst * 180 / np.pi, self.theta_lst * 180 / np.pi, self.landscape_true, 100)
        elif mode in ['minus', 'false', 2]:
            plt.contourf(self.phi_lst * 180 / np.pi, self.theta_lst * 180 / np.pi, self.landscape_false, 100)
        else:
            plt.contourf(self.phi_lst * 180 / np.pi, self.theta_lst * 180 / np.pi, self.twice_landscape, 100)
        plt.colorbar()
        plt.xlabel('Phi')
        plt.ylabel('Theta')
        plt.show()

    ''' Main program '''

    def Calculate(self, raw_p, raw_m, get_sort=True): # raw_p and raw_m must be sorted by NV axis and from the transition frequencies.
        self.raw_p, self.raw_m = raw_p, raw_m 
        self.twice_landscape = self.Digging__Twice_Lanscape()
        self.angle_prob = self.Find_Minimum_Landscape(self.twice_landscape)
        self.B_val = np.mean(self.Find_B(self.raw_p, raw_m))
        for i in range(self.angle_prob.shape[0]):
            p, m  = self.A_Transition_B(B_Vector(self.B_val, self.angle_prob[i, 0], self.angle_prob[i, 1]))
            self.Frequenices_prob.append([p, m])
        if not get_sort:
            return self.Frequenices_prob
        First_term_fre = self.SortThePrime(self.Frequenices_prob[0][0], self.Frequenices_prob[0][1], raw_p, raw_m)
        Second_term_fre = self.SortThePrime(self.Frequenices_prob[1][0], self.Frequenices_prob[1][1], raw_p, raw_m)
        return [First_term_fre, Second_term_fre]

