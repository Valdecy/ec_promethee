############################################################################

# Created by: Marcio Pereira Basilio, Valdecy Pereira, Fatih Yigit
# email:      valdecy.pereira@gmail.com
# GitHub:     <https://github.com/Valdecy>

# The EC-PROMETHEE Method - A Committee Approach for Outranking Problems Using Randoms Weights

# Citation: 
# BASILIO, M.P.; PEREIRA, V.; YIGIT, F. (2023). New Hybrid EC-Promethee Method with Multiple Iterations of Random Weight Ranges: Applied to the Choice of Policing Strategies. Mathematics. Vol. 11, Iss. 21. DOI: https://doi.org/10.3390/math11214432 

############################################################################

# Required Libraries
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns

from collections import Counter
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

###############################################################################

# EC PROMETHEE Class
class ec_promethee():
    def __init__(self, dataset, criterion_type, Q = [], S = [], P = [], F = [], custom_sets = [], iterations = 10000):
      self.data  = np.copy(dataset).astype(float)
      self.ctype = criterion_type
      self.q     = Q
      self.s     = S
      self.p     = P
      self.f     = F
      self.cset  = custom_sets
      self.iter  = iterations
      self.run()
      
    ###############################################################################
    
    # Function: CRITIC (CRiteria Importance Through Intercriteria Correlation). From https://github.com/Valdecy/pyDecision
    def critic_method(self):
        X     = np.copy(self.data).astype(float)
        best  = np.zeros(X.shape[1])
        worst = np.zeros(X.shape[1])
        for i in range(0, X.shape[1]):
            if (self.ctype[i] == 'max'):
                best[i]  = np.max(X[:, i])
                worst[i] = np.min(X[:, i])
            else:
                best[i]  = np.min(X[:, i])
                worst[i] = np.max(X[:, i])
            if (best[i] == worst[i]):
                best[i]  = best[i]  + 1e-9
                worst[i] = worst[i] - 1e-9
        for j in range(0, X.shape[1]):
            X[:,j] = ( X[:,j] - worst[j] ) / ( best[j] - worst[j] )
        std      = (np.sum((X - X.mean())**2, axis = 0)/(X.shape[0] - 1))**(1/2)
        sim_mat  = np.corrcoef(X.T)
        conflict = np.sum(1 - sim_mat, axis = 1)
        infor    = std*conflict
        weights  = infor/np.sum(infor)
        return weights

    ###############################################################################
    
    # Function: Entropy. From https://github.com/Valdecy/pyDecision
    def entropy_method(self):
        X = np.copy(self.data).astype(float)
        for j in range(0, X.shape[1]):
            if (self.ctype[j] == 'max'):
                X[:,j] =  X[:,j] / np.sum(X[:,j])
            else:
                X[:,j] = (1 / X[:,j]) / np.sum((1 / X[:,j]))
        X = np.abs(X)
        H = np.zeros((X.shape))
        for j, i in itertools.product(range(H.shape[1]), range(H.shape[0])):
            if (X[i, j]):
                H[i, j] = X[i, j] * np.log(X[i, j] + 1e-9)
        h = np.sum(H, axis = 0) * (-1 * ((np.log(H.shape[0] + 1e-9)) ** (-1)))
        d = 1 - h
        d = d + 1e-9
        w = d / (np.sum(d))
        return w

    ###############################################################################
    
    # Function: Distance Matrix. From https://github.com/Valdecy/pyDecision
    def distance_matrix(self, criteria = 0):
        distance_array = np.zeros(shape = (self.data.shape[0], self.data.shape[0]))
        for i in range(0, distance_array.shape[0]):
            for j in range(0, distance_array.shape[1]):
                distance_array[i,j] = self.data[i, criteria] - self.data[j, criteria]
        return distance_array
    
    # Function: Preferences. From https://github.com/Valdecy/pyDecision
    def preference_degree(self, W):
        pd_array = np.zeros(shape = (self.data.shape[0], self.data.shape[0]))
        for k in range(0, self.data.shape[1]):
            distance_array = self.distance_matrix(criteria = k)
            for i in range(0, distance_array.shape[0]):
                for j in range(0, distance_array.shape[1]):
                    if (i != j):
                        if (self.f[k] == 't1'):
                            if (distance_array[i,j] <= 0):
                                distance_array[i,j]  = 0
                            else:
                                distance_array[i,j] = 1
                        if (self.f[k] == 't2'):
                            if (distance_array[i,j] <= self.q[k]):
                                distance_array[i,j]  = 0
                            else:
                                distance_array[i,j] = 1
                        if (self.f[k] == 't3'):
                            if (distance_array[i,j] <= 0):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > 0 and distance_array[i,j] <= self.p[k]):
                                distance_array[i,j]  = distance_array[i,j]/self.p[k]
                            else:
                                distance_array[i,j] = 1
                        if (self.f[k] == 't4'):
                            if (distance_array[i,j] <= self.q[k]):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > self.q[k] and distance_array[i,j] <= self.p[k]):
                                distance_array[i,j]  = 0.5
                            else:
                                distance_array[i,j] = 1
                        if (self.f[k] == 't5'):
                            if (distance_array[i,j] <= self.q[k]):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > self.q[k] and distance_array[i,j] <= self.p[k]):
                                distance_array[i,j]  =  (distance_array[i,j] - self.q[k])/(self.p[k] -  self.q[k])
                            else:
                                distance_array[i,j] = 1
                        if (self.f[k] == 't6'):
                            if (distance_array[i,j] <= 0):
                                distance_array[i,j]  = 0
                            else:
                                distance_array[i,j] = 1 - math.exp(-(distance_array[i,j]**2)/(2*self.s[k]**2))
                        if (self.f[k] == 't7'):
                            if (distance_array[i,j] == 0):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > 0 and distance_array[i,j] <= self.s[k]):
                                distance_array[i,j]  =  (distance_array[i,j]/self.s[k])**0.5
                            elif (distance_array[i,j] > self.s[k] ):
                                distance_array[i,j] = 1
            pd_array = pd_array + W[k]*distance_array
        pd_array = pd_array/sum(W)
        return pd_array
    
    # Function: Rank. From https://github.com/Valdecy/pyDecision
    def ranking(self):
        flow_0  = np.arange(1, self.p2_matrix.shape[1] + 1)
        flow_1  = np.sum(self.p2_matrix, axis = 0)
        flow    = np.column_stack((flow_0, flow_1))
        flow    = flow[np.argsort(flow[:, 1])]
        flow    = flow[::-1]
        rank_xy = np.zeros((flow.shape[0], 2))
        plt.figure(figsize = (10, 10), dpi = 100)
        for i in range(0, rank_xy.shape[0]):
            rank_xy[i, 0] = 0
            rank_xy[i, 1] = flow.shape[0]-i
        for i in range(0, rank_xy.shape[0]):
            if (flow[i,1] >= 0):
                plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
            else:
                plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))
        for i in range(0, rank_xy.shape[0]-1):
            plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
        axes = plt.gca()
        axes.set_xlim([-1, +1])
        ymin = np.amin(rank_xy[:,1])
        ymax = np.amax(rank_xy[:,1])
        if (ymin < ymax):
            axes.set_ylim([ymin, ymax])
        else:
            axes.set_ylim([ymin-1, ymax+1])
        plt.axis('off')
        plt.show()
        return
    
    # Function: Promethee II. From https://github.com/Valdecy/pyDecision
    def promethee_ii(self, W):
        pd_matrix  = self.preference_degree(W)
        flow_plus  = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
        flow_minus = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
        flow       = flow_plus - flow_minus
        flow       = np.reshape(flow, (pd_matrix.shape[0], 1))
        flow       = np.insert(flow, 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
        return flow
    
    ###############################################################################

    # Function: Generate Ranks. From https://github.com/Valdecy/pyDecision
    def generate_rank_array(self, arr, sorted_indices):
        rank_array = np.zeros(len(arr), dtype = int)
        for rank, index in enumerate(sorted_indices, start = 1):
            rank_array[index] = rank
        return rank_array
    
    # Function: Find Mode
    def find_column_modes(self, matrix):
        transposed_matrix = np.transpose(matrix)
        mode_list         = []
        for column in transposed_matrix:
            counter   = Counter(column)
            max_count = max(counter.values())
            modes     = [x for x, count in counter.items() if count == max_count]
            mode_list.append(modes)
        return mode_list

    # Function: Tranpose Dictionary. From https://github.com/Valdecy/pyDecision
    def transpose_dict(self, rank_count_dict):
        transposed_dict = {}
        list_length     = len(next(iter(rank_count_dict.values())))
        for i in range(list_length):
            transposed_dict[i+1] = [values[i] for values in rank_count_dict.values()]
        return transposed_dict

    # Function: Plot Ranks. Adapted From https://github.com/Valdecy/pyDecision
    def plot_rank_freq(self, size_x = 8, size_y = 10):
        flag_1             = 0
        ranks              = self.ranks_matrix.T
        alternative_labels = [f'a{i+1}' for i in range(ranks.shape[0])]
        rank_count_dict    = {i+1: [0]*ranks.shape[0] for i in range(0, ranks.shape[0])}
        for i in range(0, ranks.shape[0]):
            for j in range(0, ranks.shape[1]):
                rank = int(ranks[i, j])
                rank_count_dict[i+1][rank-1] = rank_count_dict[i+1][rank-1] + 1
        rank_count_dict = self.transpose_dict(rank_count_dict)
        fig, ax         = plt.subplots(figsize = (size_x, size_y))
        try:
          cmap   = colormaps.get_cmap('tab20')
          colors = [cmap(i) for i in np.linspace(0, 1, ranks.shape[0])]
        except:
          colors = plt.cm.get_cmap('tab20', ranks.shape[0])
          flag_1 = 1
        bottom = np.zeros(len(alternative_labels))
        for rank, counts in rank_count_dict.items():
            if (flag_1 == 0):
              bars = ax.barh(alternative_labels, counts, left = bottom, color = colors[rank-1])
            else:
              bars = ax.barh(alternative_labels, counts, left = bottom, color = colors(rank-1))
            bottom = bottom + counts
            for rect, c in zip(bars, counts):
                if (c > 0):
                    width = rect.get_width()
                    ax.text(width/2 + rect.get_x(), rect.get_y() + rect.get_height() / 2, f"r{rank} ({c})", ha = 'center', va = 'center', color = 'black')
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(integer = True))
        ax.tick_params(axis = 'y', which = 'both', pad = 25)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Alternative')
        ax.set_title('Rank Frequency per Alternative')
        plt.show()
        return

    # Function: Normalized Weights Box Plot
    def wm_boxplot(self, size_x = 15, size_y = 7):
        plt.figure(figsize = (size_x, size_y))
        df_melted = self.df_w.melt(var_name = 'Columns', value_name = 'Values')
        sns.boxplot(x = 'Columns', y = 'Values', data = df_melted)
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.show()
        
    # Function: PROMETHEE II Box Plot
    def p_ii_boxplot(self, size_x = 15, size_y = 7):
        plt.figure(figsize = (size_x, size_y))
        df_melted = self.df_p.melt(var_name = 'Columns', value_name = 'Values')
        sns.boxplot(x = 'Columns', y = 'Values', data = df_melted)
        plt.axhline(0, color = 'red', linestyle = '--')
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.show()

    # Function: EC PROMETHEE
    def run(self):
        X                    = np.copy(self.data).astype(float)
        min_indices          = np.where(np.array(self.ctype) == 'min')[0]
        X[:, min_indices]    = 1.0 / X[:, min_indices]
        self.critic_weights  = self.critic_method()
        self.entropy_weights = self.entropy_method()
        self.ranks_matrix    = []
        self.wnorm_matrix    = []
        self.p2_matrix       = []
        self.sol             = []
        lower_upper_pairs    = []
        #print ('Entropy Weights:')
        #formatted         = ['{:.3f}'.format(val) for val in self.entropy_weights ]
        #$print('[' + ', '.join(formatted) + ']')
        for i in range(len(self.critic_weights)):
            all_weights = [self.entropy_weights[i], self.critic_weights[i]]
            if (self.cset):
                for custom_set in self.cset:
                    if (i < len(custom_set)):
                        all_weights.append(custom_set[i])
            lower = min(all_weights)
            lower = max(1e-10, lower)
            upper = max(all_weights)
            lower_upper_pairs.append((lower, upper))
        weights_data = []
        weights_data.append(['Entropy'] + [self.entropy_weights[i]  for i in range(len(self.entropy_weights))])
        weights_data.append(['Critic']  + [self.critic_weights[i] for i in range(len(self.critic_weights))])
        if (self.cset):
            count = 1
            for custom_set in self.cset:
                weights_data.append(['Custom Weights ' + str(count)] + [custom_set[i] for i in range(len(custom_set))])
                count = count + 1
        lower_weights   = ['Lower'] + [lower for lower, upper in lower_upper_pairs]
        upper_weights   = ['Upper'] + [upper for lower, upper in lower_upper_pairs]
        weights_data.append(lower_weights)
        weights_data.append(upper_weights)
        columns         = ['Weight Name'] + ['g'+str(i+1) for i in range(len(self.critic_weights))]
        self.weights_df = pd.DataFrame(weights_data, columns = columns)
        self.weights_df.set_index('Weight Name', inplace = True)
        for _ in range(self.iter):
            random_weights   = np.array([random.uniform(lower, upper) for lower, upper in lower_upper_pairs])
            #random_weights   = random_weights / np.sum(random_weights)
            self.wnorm_matrix.append(random_weights)
            promethee_result = self.promethee_ii(random_weights)
            self.p2_matrix.append(promethee_result[:,-1])
            ranks            = np.argsort(promethee_result[:, 1])[::-1]
            ranks            = self.generate_rank_array(promethee_result[:, 1], ranks)
            self.ranks_matrix.append(ranks)
        self.wnorm_matrix = np.array(self.wnorm_matrix)
        self.ranks_matrix = np.array(self.ranks_matrix)
        self.p2_matrix    = np.array(self.p2_matrix)
        self.sol_m = self.find_column_modes(self.ranks_matrix)
        p2_sum     = np.sum(self.p2_matrix, axis = 0)
        p2_rank    = np.argsort(p2_sum)[::-1]
        p2_rank    = self.generate_rank_array(p2_sum, p2_rank)
        self.sol   = [ [item] for item in p2_rank]
        self.df_w  = pd.DataFrame(self.wnorm_matrix, columns = [f'g{i+1}' for i in range(self.wnorm_matrix.shape[1])], index = [f'Iteration {i+1}' for i in range(self.wnorm_matrix.shape[0])])
        self.df_r  = pd.DataFrame(self.ranks_matrix, columns = [f'a{i+1}' for i in range(self.ranks_matrix.shape[1])], index = [f'Iteration {i+1}' for i in range(self.ranks_matrix.shape[0])])
        self.df_p  = pd.DataFrame(self.p2_matrix, columns = [f'a{i+1}' for i in range(self.p2_matrix.shape[1])],       index = [f'Iteration {i+1}' for i in range(self.p2_matrix.shape[0])])
        return

###############################################################################