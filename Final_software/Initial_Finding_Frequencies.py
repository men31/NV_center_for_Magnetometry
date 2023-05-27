from sklearn import cluster
import numpy as np

# K-Means clustering
class Initial_Frequencies:
    def __init__(self):
        self.x = None
        self.y = None
        self.cutOn = None
        self.numCen = None
        
        # Method output parameters
        self.x_cut, self.y_cut = None, None
        self.kmean = None
        self.centroids = None
        self.completed_frequencies_centers = None
        
        # Defualt parameters
        self.perNV = 0.990
        self.find_group = False
        
    def ScopeTheData(self):
        idx_cuton = np.where(self.y <= self.cutOn)
        return self.x[idx_cuton], self.y[idx_cuton]
    
    def FindGroup_Centers(self):
        # Prepar the x, y before using K-means
        self.x_cut, self.y_cut = self.ScopeTheData()
        work_group = np.vstack((self.x_cut, self.y_cut)).T
        
        # Use K-means Algorithms
        self.kmean = cluster.KMeans(n_clusters=self.numCen)
        self.kmean.fit(work_group)
        
        # K-means output
        self.centroids = self.kmean.cluster_centers_
        self.find_group = True
        
    def Get_GroupLowest(self):
        if not self.find_group:
            self.FindGroup_Centers()
        result_label = self.kmean.labels_.astype(int)
        uqe = np.unique(result_label)
        lowest_y = []
        for i in uqe:
            y_choose = self.y_cut[np.where(result_label == i)]
            lowest_y.append(y_choose.min())
        return np.array(lowest_y)
        
    def Get_AllCenters(self):
        if self.numCen == 8:
            all_fre_cens = self.centroids[:, 0]
        else:
            lowest_pl = self.Get_GroupLowest()
            nums_fre = np.round((1 - lowest_pl) / (1 - self.perNV)).astype(int)
            result_label = self.kmean.labels_.astype(int)
            all_fre_cens = np.array([])
            for fre_idx, num in enumerate(nums_fre):
                if num in (0, 1):
                    all_fre_cens = np.append(all_fre_cens, self.centroids[:, 0][fre_idx])
                    continue
                y_choose = self.y_cut[np.where(result_label == fre_idx)]
                x_choose = self.x_cut[np.where(result_label == fre_idx)]
                work_choose = np.vstack((x_choose, y_choose)).T
                kmean = cluster.KMeans(n_clusters=num)
                kmean.fit(work_choose)
                fre_centers = kmean.cluster_centers_
                all_fre_cens = np.append(all_fre_cens, fre_centers[:, 0])
            all_fre_cens = self.Check_Output(all_fre_cens)
        return all_fre_cens
    
    def Check_Output(self, x_predict):
        work_group = np.vstack((self.x_cut, self.y_cut)).T
        # Use K-means Algorithms
        kmean = cluster.KMeans(n_clusters=8)
        kmean.fit(work_group)
        x_result = kmean.cluster_centers_[:, 0]
        idx_arr = np.arange(len(x_result))
        zero = np.zeros((len(x_result)))
        for pre in x_predict:
            min_idx = np.argmin(abs(pre - x_result))
            zero[min_idx] = pre
            idx_arr = np.delete(idx_arr, np.where(idx_arr == min_idx))
        if idx_arr.shape != 0:
            zero[idx_arr] = x_result[idx_arr]
        return zero
    
    def Run_Dogmatic(self, x, y, cutOn, numCen):
        self.x = x
        self.y = y
        self.cutOn = cutOn
        self.numCen = numCen
        self.FindGroup_Centers()
        self.completed_frequencies_centers = self.Get_AllCenters()
        return self.completed_frequencies_centers

    def Run_Simple(self, x, y, cutOn, numCen=8):
        self.x = x
        self.y = y
        self.cutOn = cutOn
        self.numCen = numCen
        self.FindGroup_Centers()
        self.completed_frequencies_centers = self.centroids[:, 0]
        return self.completed_frequencies_centers
