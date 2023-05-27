import numpy as np

from NV_Hamiltonian import *
from NV_Utils import *
from Swarm_Utils import *


class Op_MagneticVector:
    def __init__(self):
        self.NV_H = NV_Hamiltonian()
        self.m, self.p = None, None
        self.Trans = None

        # Define parameters
        self.lowbound = np.array([0, 0, -np.pi])
        self.upbound = np.array([0.1, np.pi, np.pi])
        self.maxIter = 500

        # Define output
        self.B_val, self.theta, self.phi = 0, 0, 0

    @staticmethod
    def Cost_function(actual, result):
        return np.mean((actual - result)**2)

    def ObjectiveFunction(self, X):  # X [B, theta, phi]
        B_vec = X[0] * np.array([[np.sin(X[1]) * np.cos(X[2])],
                                 [np.sin(X[1]) * np.sin(X[2])],
                                 [np.cos(X[1])]])
        p_prime, m_prime = self.NV_H.A_Transition_B(B_vec)
        Trans_prime = np.append(m_prime, p_prime)
        loss = self.Cost_function(self.Trans.copy(), Trans_prime)
        return loss

    def Calculate(self, p, m):
        self.m, self.p = m, p
        self.Trans = np.append(self.m, self.p)
        
        self.PSO = PSO(self.ObjectiveFunction, self.lowbound,
                       self.upbound, self.maxIter)
        GBEST_X, GBEST_O = self.PSO.Calculate()
        [self.B_val, self.theta, self.phi] = GBEST_X
        return np.array([self.B_val, self.theta, self.phi])
