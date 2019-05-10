import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
import scipy as sp
from scipy import signal

# open file, make sure file name extension is  .csv
cwd_agg_sub1 = "E:\\Queens\ECE\\2018 FALL\\ELEC879\\Project\\EMG Physical Action Data Set\\sub1\\Aggressive\\txt"
filename_agg_sub1 = cwd_agg_sub1 + "\\Combined.txt"
dataFrame_agg_sub1 = pd.read_csv(filename_agg_sub1, sep='\t')

cwd_nor_sub1 = "E:\\Queens\ECE\\2018 FALL\\ELEC879\\Project\\EMG Physical Action Data Set\\sub1\\Normal\\txt"
filename_nor_sub1 = cwd_nor_sub1 + "\\Combined.txt"
dataFrame_nor_sub1 = pd.read_csv(filename_nor_sub1, sep='\t')

# cwd_agg_sub3 = "E:\\Queens\ECE\\2018 FALL\\ELEC879\\Project\\EMG Physical Action Data Set\\sub3\\Aggressive\\txt"
# filename_agg_sub3 = cwd_agg_sub3 + "\\Combined.txt"
# dataFrame_agg_sub3 = pd.read_csv(filename_agg_sub3, sep='\t')

# make list of list
data_list_agg_sub1 = dataFrame_agg_sub1.values.tolist()
list_of_list_agg_sub1 = np.transpose(data_list_agg_sub1)

data_list_nor_sub1 = dataFrame_nor_sub1.values.tolist()
list_of_list_nor_sub1 = np.transpose(data_list_nor_sub1)

# data_list_agg_sub3 = dataFrame_agg_sub3.values.tolist()
# list_of_list_agg_sub3 = np.transpose(data_list_agg_sub3)

# print list of list
print(list_of_list_agg_sub1)
print(list_of_list_agg_sub1.shape)

print(list_of_list_nor_sub1)
print(list_of_list_nor_sub1.shape)

# print(list_of_list_agg_sub3)
# print(list_of_list_agg_sub3.shape)

#
#设置坐标轴刻度
my_x_ticks = np.arange(0, 110000, 1000)
my_y_ticks = np.arange(-4000,4200, 500)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('samples')
plt.ylabel('no idea')

Fig_agg_sub1= list_of_list_agg_sub1[0]
Fig_nor_sub1= list_of_list_nor_sub1[0]
print(Fig_agg_sub1)
# print(Fig_nor_sub1)

# Fig_agg_sub1_mf = sp.signal.medfilt(list_of_list_agg_sub1[0], 101)
plt.plot(Fig_agg_sub1)
plt.plot(Fig_agg_sub1_mf)
# plt.plot(Fig_nor_sub1)
plt.show()

