from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

from Sorting_Raw_Frequencies import Sorting_RawFre
from Swarm_Optimize_MagneticVector import Op_MagneticVector

class Find_Mag_GUI:
    def __init__(self) -> None:
        self.df = None
        self.fre_sort_byNV = None

    @staticmethod
    def Call_csvfiles(csv_path):
        name_csv = [i for i in os.listdir(csv_path) if i.endswith('csv')]
        CSV_dict = {}
        for name in name_csv:
            CSV_dict[name] = pd.read_csv(csv_path + name)
        return CSV_dict

    @staticmethod
    def Check_KeyboardInput(key_input):
        try:
            key_input = int(key_input)
        except ValueError:
            key_input = str(key_input)
        return key_input

    def Get_SortedFre(self):
        try:
            return np.array([self.df[(self.df['NV_label'] == uqe)]['Fitted_frequncies (Hz)'].to_numpy() for uqe in np.sort(self.df['NV_label'].unique())])
        except Exception:
            return np.array([self.df[(self.df['NV_label'] == uqe)]['Initial_frequncies (Hz)'].to_numpy() for uqe in np.sort(self.df['NV_label'].unique())])

    def Plot_GUI(self):

        # Caculating to find the B, theta, phi
        start_time = time.time()
        self.fre_sort_byNV = self.Get_SortedFre()
        transition_prob = Sorting_RawFre().Calculate(self.fre_sort_byNV[:,0]/1e9, self.fre_sort_byNV[:,1]/1e9)
        B_result_params_1 = Op_MagneticVector().Calculate(transition_prob[0][0], transition_prob[0][1])
        B_result_params_2 = Op_MagneticVector().Calculate(transition_prob[1][0], transition_prob[1][1])
        end_time = time.time()
        print('If theta less than 90 degree')
        print('Result')
        print('B: ', B_result_params_1[0])
        print('Theta: ', B_result_params_1[1] * 180 / np.pi)
        print('Phi: ', B_result_params_1[2] * 180 / np.pi)
        print('If theta more than 90 degree')
        print('Result')
        print('B: ', B_result_params_2[0])
        print('Theta: ', B_result_params_2[1] * 180 / np.pi)
        print('Phi: ', B_result_params_2[2] * 180 / np.pi)

        print(f'Process time: {end_time - start_time} s')
        _ = input('Press ENTER to continue: ')

    def Calculate(self, fre_csv_path):
        while True:
            self.csv_path = fre_csv_path
            CSV_Files = self.Call_csvfiles(self.csv_path)
            key_lst = list(CSV_Files.keys())
            for idx, key in enumerate(key_lst):
                print(f'[{idx+1}] {key}')
            print('[r] Restart')
            print('[q] Quit')
            
            keyboard_input = self.Check_KeyboardInput(input('Please Enter: ').lower())
            if keyboard_input == 'q':
                print('Quit')
                break
            elif keyboard_input == 'r':
                continue
            elif keyboard_input in np.arange(len(key_lst) + 1):
                working_csv = CSV_Files[key_lst[keyboard_input-1]]
                self.working_name = key_lst[keyboard_input-1].split('.csv')
                print(f'Number: {keyboard_input} Name: {key_lst[keyboard_input-1]}')
            else:
                print('[-] Can not find your input number or command, please enter again')
        
            self.df = working_csv
            self.Plot_GUI()

if __name__ == '__main__':
    fre_csv_path = './Frequencies_CSV_file/'
    Find_Mag_GUI().Calculate(fre_csv_path=fre_csv_path)
