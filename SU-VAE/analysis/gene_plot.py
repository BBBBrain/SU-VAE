"""2023 / 8 / 18
8: 42
dell"""

"""2023 / 8 / 10
10: 26
dell
本地计算机实验
"""

"""2023 / 7 / 22
8: 32
dell
偏最小二乘方法分析  基因差异矩阵和  脑功能连接shared、specific矩阵之间的相关性，并找出较高相关水平的基因
"""
import os
import scipy.io as scio
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils import resample
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from scipy.stats import rankdata, linregress
from scipy import stats



def _calculate_vips(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T,t),q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(np.matmul(s.T, weight))/total_s)
    return vips


z_s = StandardScaler()

# mpl.use('Agg')
id = []
X_L = []
X_R = []
Y = []

# gene_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA23_gene_func_matrix\\"   # BA23
gene_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA28_gene_func_matrix\\"   # BA28
# gene_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\charm_gene_func_matrix\\"   # charm
gene_list_dir = np.loadtxt("G:\\IPMI_Jornal\\gene_data\\gene_data\\gene_id_list.txt",dtype=np.int64)
# print(gene_list_dir)

for g_id in sorted(gene_list_dir):
    g_id = str(g_id)
    L_gene_path = gene_dir + g_id + "_L.mat"
    R_gene_path = gene_dir + g_id + "_R.mat"
    L_gene = scio.loadmat(L_gene_path)['data']
    # print('L_gene',L_gene.shape)
    R_gene = scio.loadmat(R_gene_path)['data']
    uptri_idL = np.triu_indices_from(L_gene, k=1)  # 返回矩阵上三角元素(k=1,不包含对角元素)的索引index
    uptri_L = L_gene[(uptri_idL)]  # 将index返回矩阵，矩阵返回对应index的值
    uptri_idR = np.triu_indices_from(R_gene, k=1)  # 返回矩阵上三角元素(k=1,不包含对角元素)的索引index
    uptri_R = R_gene[(uptri_idR)]  # 将index返回矩阵，矩阵返回对应index的值
    X_L.append(uptri_L)
    X_R.append(uptri_R)
    # X_L.append(L_gene.flatten())
    # X_R.append(R_gene.flatten())

X_L = np.array(X_L).transpose(1,0)
X_R = np.array(X_R).transpose(1,0)
# X_L = z_s.fit_transform(X_L)
# X_R = z_s.fit_transform(X_R)

# print(X_L.shape)
# print(X_R.shape)
#
# Define PLS object
"""
BA23   10
"""
# h_sh_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA23_average_human_shared.jpg"  # BA23 L: 3     shared:L:4
# h_sp_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA23_average_human_specific.jpg"  # BA23 L:3        shared:L:4
"""
BA28  16
"""
h_sh_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA28_average_human_shared.jpg" #BA28  L:8           shared:  L:13
h_sp_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\average_human.jpg" #BA28  L:8           shared:  L:13
"""
charm 21
"""
# h_sh_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\average_human_shared.jpg"  # charm  L: 11     shared:  L:16
# h_sp_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\average_human_specific.jpg"  # charm  L: 11   shared:  L:16

k = 28

adjacency_matrix = cv2.imread(h_sh_dir)
adjacency_matrix = adjacency_matrix[:,:,0]/255
# adjacency_matrix = z_s.fit_transform(adjacency_matrix)
Y = adjacency_matrix.reshape(k,k)
uptri_idY = np.triu_indices_from(Y, k=1)  # 返回矩阵上三角元素(k=1,不包含对角元素)的索引index
Y_sh = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
Y_sh2 = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
adjacency_matrix = cv2.imread(h_sp_dir)
adjacency_matrix = adjacency_matrix[:,:,0]/255
# adjacency_matrix = z_s.fit_transform(adjacency_matrix)
Y = adjacency_matrix.reshape(k,k)
uptri_idY = np.triu_indices_from(Y, k=1)  # 返回矩阵上三角元素(k=1,不包含对角元素)的索引index
Y_sp = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
Y_sp2 = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
# print(len(Y))
# print(len(Y))

# Y_sh = Y_sh - Y_sp2
# # Y_sh[Y_sh < 0] = 0
Y_sp = Y_sp - Y_sh2
# # Y_sp[Y_sp < 0] = 0
# # Y = np.column_stack((Y_sh, Y_sp))
X = X_L
# X = z_s.fit_transform(X)
# print(Y_sh.shape)
# print(X.shape)
# print(Y_sh,Y_sp)

# Y_sh = z_s.fit_transform(Y_sh)
# Y_sp = z_s.fit_transform(Y_sp)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# print(X)

"""
寻找最佳ncom
"""
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# mse = []
# for i in np.arange(2, 40):
#     pls = PLSRegression(n_components=i)
#     score = -1*model_selection.cross_val_score(pls, X, Y_sp, cv=cv,
#                scoring='neg_mean_squared_error').mean()
#     mse.append(score)
# for i in range(2,40):
#     print(i,mse[i-2])

from sklearn.cross_decomposition import PLSRegression
#
# Y_sp  = Y_sp
n_components = 22

#
pls = PLSRegression(n_components=n_components)
pls.fit(X, Y_sp)

#




# print(X.shape)
# print(X)
# print(Y_sp.shape)
# print(Y_sp)
#
# # component_variances = np.var(pls.x_scores_, axis=0)
# # # 计算每个成分方差的比例
# # variance_ratios = component_variances / np.sum(component_variances)
# # # 打印每个成分的方差解释率
# # for i, ratio in enumerate(variance_ratios):
# #     component_number = i + 1
# #     print(f'Component {component_number}: {ratio}')
# # # explained_variance = np.var(X - pls.predict(X), axis=0) / np.var(X, axis=0)
# # # print("ratio",explained_variance)
#
# #
# #
# # #
# # X_scores = pls.x_scores_[:,0]
# # print(X_scores)
# # X_scores = pls.x_scores_[:,1]
# # print(X_scores)
# X_scores = pls.transform(X)[:, 0]
#
# m = X_scores.max()
# s = X_scores.min()
# X_scores = (X_scores - s)/(m-s)
# # Y_variable = Y_sp
# Y_variable = pls.predict(X)
#
# # print(X_scores.shape)
# # print(Y_variable.shape)
#
#
# # plt.scatter(X_scores, Y_sh, alpha=0.5)
# # plt.xlabel('Actual Output')
# # plt.ylabel('PLSR Scores')
# # plt.title('Correlation between Actual Output and PLSR Scores')
# # correlation = np.corrcoef(Y_variable.reshape(-1), X_scores.reshape(-1))[0, 1]
# # plt.text(0.1, 0.9, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)
# #
# # plt.show()
# #
# #
# slope, intercept, r_value, p_value, std_err = linregress(X_scores, Y_variable.reshape(-1))
# p_values = np.array([p_value])
# rejected, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
#
# font = {'family': 'Times New Roman'}
# plt.rc('font', **font)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# # 绘制散点图和线性拟合线
# plt.scatter(X_scores, Y_sp, label='Data',s=10,c="#B4EEB4")
# plt.plot(X_scores, intercept + slope * X_scores, color='red', label='Linear Fit')
# plt.xlabel('Gene PLSR1 score',fontsize=15)
# plt.ylabel('Human-specifc FC',fontsize=15)
# # plt.legend()
# # plt.title('Linear Fit of PLSR Component Scores vs. Y Variable')
#
# plt.text(-100, 0.08, f"r = {r_value:.4f}\n p < {corrected_p_values[0]:.2e}", fontsize=13, fontdict={'style': 'italic'},color='black')
# #plt.text(-100, 0.08,f"$r = {r_value:.4f}$\n$FDR$-corrected $p = {corrected_p_values:.2e}, fontsize=13, fontdict={'style': 'italic'},color='black'$")
# plt.show()
#
# # 输出线性回归结果
# print("Slope:", slope)
# print("Intercept:", intercept)
# print("R-value (Correlation Coefficient):", r_value)
# print("P-value:", p_value)
# print("Standard Error:", std_err)













































