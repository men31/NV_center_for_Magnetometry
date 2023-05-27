import numpy as np
import matplotlib.pyplot as plt


def B_Vector(B_val ,theta, phi):
    return B_val * np.array([[np.cos(phi) * np.sin(theta)], [np.sin(phi) * np.sin(theta)], [np.cos(theta)]])

def Find_theta(vec, vecZ):
    return np.arccos(np.dot(vec.T, vecZ) / (np.linalg.norm(vec) * np.linalg.norm(vecZ)))

def Find_phi(vec, vecX, vecY):
    vec_ = np.array(
        [[np.dot(vec.T, vecX)], [np.dot(vec.T, vecY)]]).reshape(2, 1)
    x_2d, y_2d = np.array([[1], [0]]), np.array([[0], [1]])
    if np.linalg.norm(vec_) == 0:
        phi = 0
    else:
        phi = np.arccos(np.dot(vec_.T, x_2d) /
                    abs(np.linalg.norm(vec_) * np.linalg.norm(x_2d)))
    if np.dot(vec_.T, y_2d) < 0:
        phi = - phi
    return phi

def Rotate_Theta(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]], dtype=object)

def Rotate_Phi(phi):
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]], dtype=object)

def Rotate_Sphere(theta, phi):
    rotTheta = Rotate_Theta(theta)
    rotPhi = Rotate_Phi(phi)
    return rotPhi @ rotTheta

def Make_B_Probability(All_alpha, u_arr_ori): # u = 4 x 3
        init_Bprob = []
        for idx, u in enumerate(u_arr_ori):
            theta_adjust = np.sum(Find_theta(u, np.array([[0], [0], [1]])))
            phi_adjust = np.sum(Find_phi(u, np.array([[1], [0], [0]]), np.array([[0], [1], [0]])))
            B_prob = B_Vector(1, All_alpha[idx], 0)
            B_prob = Rotate_Sphere(theta_adjust, phi_adjust) @ B_prob
            init_Bprob.append(B_prob)
        return np.array(init_Bprob)

def RotateVecAround(centerVec, rotVec, omega):
    return np.cos(omega) * rotVec + np.cross(centerVec.T, rotVec.T).T * np.sin(omega) + \
            centerVec * np.dot(centerVec.T, rotVec) * (1 - np.cos(omega))


