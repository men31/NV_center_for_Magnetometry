import cv2
import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt


'''Use to show all mat file'''


def MatShow(mat_dict: dict):
    lst = list(mat_dict.keys())
    for i in range(len(lst)):
        mat_val = mat_dict[lst[i]]
        plt.figure(i)
        plt.imshow(mat_val)
        plt.title(lst[i])
        plt.show()


''' Use for Call one Mat file '''


class MatData:

    def __init__(self, name, pl, xvals, sig, ref, ODESR_label):
        self.name = name
        self.pl = pl.reshape(-1)
        self.xvals = xvals.reshape(-1)
        self.sig = sig
        self.ref = ref
        self.ODESR_label = ODESR_label


class All_MatData:

    def __init__(self, mat_path):
        self.mat_path = mat_path

    def readMat(self):
        all_lst = os.listdir(self.mat_path)
        return [i for i in all_lst if i.endswith('mat')]

    def CallMatData(self):
        name_lst = self.readMat()
        Mat_dict = {}
        for i in name_lst:
            path_now = self.mat_path + i
            file = sio.loadmat(path_now)
            pl, xvals, sig, ref, ODESR_label = file['pl'], file['xvals'], file['sig'], file['ref'], file['ODESR_label']
            Mat_dict[i] = MatData(i, pl, xvals, sig, ref, ODESR_label[0])
        return Mat_dict



