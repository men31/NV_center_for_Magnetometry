from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

if __name__ == '__main__':
    u1_ori = np.array([[1], [1], [1]]) / 3 ** (1 / 2)
    u2_ori = np.array([[1], [-1], [-1]]) / 3 ** (1 / 2)
    u3_ori= np.array([[-1], [-1], [1]]) / 3 ** (1 / 2)
    u4_ori = np.array([[-1], [1], [-1]]) / 3 ** (1 / 2)
    u_arr = np.array((u1_ori, u2_ori, u3_ori, u4_ori)).reshape(4, 3).T

    color_lst = ['maroon', 'orange', 'royalblue', 'lime', 'cyan', 'yellow', 'darkviolet', 'hotpink']

    fig = plt.figure('NV Structure')
    ax = plt.axes(projection='3d')
    for i in range(u_arr.shape[1]):
        ax.plot3D([0, u_arr[0, i]], [0, u_arr[1, i]], [0, u_arr[2, i]], lw=3, color=color_lst[i], alpha=0.5, label=f'NV #{i+1}')
        ax.scatter3D(u_arr[0, i], u_arr[1, i], u_arr[2, i], color=color_lst[i], s=150)
    ax.scatter3D(0, 0, 0, color='black', s=150)
    plt.title('NV axis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()