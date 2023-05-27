import numpy as np

''' Sub program for every class'''


def find_theta(vec, vecZ):
    return np.arccos(np.dot(vec.T, vecZ) / (np.linalg.norm(vec) * np.linalg.norm(vecZ)))


def find_phi(vec, vecX, vecY):
    vec_ = np.array(
        [[np.dot(vec.T, vecX)], [np.dot(vec.T, vecY)]]).reshape(2, 1)
    x_2d, y_2d = np.array([[1], [0]]), np.array([[0], [1]])
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


def MakeCoordinate_u(u):
    n_axs = np.eye((3))
    theta = find_theta(u, n_axs[:, 2].reshape(-1, 1))
    phi = find_phi(u, n_axs[:, 0].reshape(-1, 1), n_axs[:, 1].reshape(-1, 1))
    rot = Rotate_Sphere(sum(theta), sum(phi)).astype(np.float64)
    u_axs = (rot @ n_axs)
    return u_axs.T


def Projection_Matrix(Parallel_Axs):
    dim0 = Parallel_Axs.shape[0]
    Projection_Vec = Parallel_Axs @ Parallel_Axs.reshape(dim0, 1, 3)
    Projection_Val = Parallel_Axs.reshape(dim0, 1, 3) @ Parallel_Axs
    return Projection_Vec / Projection_Val


def GenerateOrthoAxis(Ori_Axs, num):
    rand_axs = np.random.rand(num, 3, 1)
    Ortho_lst = [Ori_Axs]
    for idx in range(num):
        RandAxs = rand_axs[idx].copy()
        for j in range(idx+1):
            Parallel_Axs = Ortho_lst[j]
            Projection_UnitVec = Projection_Matrix(Parallel_Axs)
            Parallel_Vec = np.dot(Projection_UnitVec, rand_axs[idx].copy())
            RandAxs = RandAxs - Parallel_Vec
        Ortho_lst.append(
            RandAxs / np.sqrt(RandAxs.reshape(Ori_Axs.shape[0], 1, 3) @ RandAxs))
    return Ortho_lst


def MakeCoordinate_u_V2(Ortho_arr):
    for i in range(len(Ortho_arr)):
        if i == 0:
            NV_axs = Ortho_arr[0]
        else:
            NV_axs = np.hstack((NV_axs, Ortho_arr[i]))
    return NV_axs.reshape(4, len(Ortho_arr), 3)[:, ::-1]
