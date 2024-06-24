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
from scipy.stats import rankdata
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

mpl.use('Agg')
id = []
X_L = []
X_R = []
Y = []

# gene_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA23_gene_func_matrix\\"   # BA23
gene_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA28_gene_func_matrix\\"   # BA28
# gene_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\charm_gene_func_matrix\\"   # charm
gene_list_dir = np.loadtxt("G:\\IPMI_Jornal\\gene_data\\gene_data\\gene_id_list2.txt",dtype=np.int64)
# print(gene_list_dir)

# for g_id in sorted(gene_list_dir):
for g_id in gene_list_dir:
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
h_sp_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\BA28_average_human_specific.jpg" #BA28  L:8           shared:  L:13
"""
charm 21
"""
# h_sh_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\average_human_shared.jpg"  # charm  L: 11     shared:  L:16
# h_sp_dir = "G:\\IPMI_Jornal\\gene_data\\gene_data\\average_human_specific.jpg"  # charm  L: 11   shared:  L:16

k = 28

adjacency_matrix = cv2.imread(h_sh_dir)
adjacency_matrix = adjacency_matrix[:,:,0]
# adjacency_matrix = z_s.fit_transform(adjacency_matrix)
Y = adjacency_matrix.reshape(k,k)
uptri_idY = np.triu_indices_from(Y, k=1)  # 返回矩阵上三角元素(k=1,不包含对角元素)的索引index
Y_sh = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
Y_sh2 = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
adjacency_matrix = cv2.imread(h_sp_dir)
adjacency_matrix = adjacency_matrix[:,:,0]
# adjacency_matrix = z_s.fit_transform(adjacency_matrix)
Y = adjacency_matrix.reshape(k,k)
uptri_idY = np.triu_indices_from(Y, k=1)  # 返回矩阵上三角元素(k=1,不包含对角元素)的索引index
Y_sp = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
Y_sp2 = Y[(uptri_idY)]  # 将index返回矩阵，矩阵返回对应index的值
# print(len(Y))

# Y_sh = Y_sh - Y_sp2
# # Y_sh[Y_sh < 0] = 0
# Y_sp = Y_sp - Y_sh2
# # Y_sp[Y_sp < 0] = 0
# # Y = np.column_stack((Y_sh, Y_sp))
X = X_L
# print(Y_sh.shape)
# print(X.shape)
# print(Y_sh,Y_sp)

# Y_sh = z_s.fit_transform(Y_sh)
# Y_sp = z_s.fit_transform(Y_sp)

"""
寻找最佳ncom
"""
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# mse = []
# for i in np.arange(2, 30):
#     pls = PLSRegression(n_components=i)
#     score = -1*model_selection.cross_val_score(pls, scale(X), Y_sh, cv=cv,
#                scoring='neg_mean_squared_error').mean()
#     mse.append(score)
# for i in range(2,30):
#     print(i,mse[i-2])


#
# Y_sp  = Y_sp
comp = 2

#
pls = PLSRegression(n_components=comp)
pls.fit_transform(X, Y_sp)
print("sadsad",pls.coef_.shape)
# pls_weights_sp = pls.coef_.flatten()  # 获取每个自变量的PLS权重
# pls_weights_sp = pls.coef_[:,0]  # 获取每个自变量的PLS权重
# print(pls_weights_sp.shape)
# pls_weights_sp = pls.coef_[:,0]  # 获取每个自变量的PLS权重
pls_weights_sp2 = pls.x_loadings_[:,0]  # 获取每个自变量的PLS权重
print(pls_weights_sp2.shape)
print(pls.x_weights_.shape)
# pls = PLSRegression(n_components=13)
# pls.fit(X, Y_sh)
# pls_weights_sh = pls.coef_.flatten()  # 获取每个自变量的PLS权重


# explained_variance = np.var(pls.x_scores_, axis=0) / np.var(X, axis=0)
#
# print("Explained Variance for Each Component:")
# for i, ev in enumerate(explained_variance):
#     print(f"Component {i+1}: {ev:.4f}")


n_bootstrap = 1000  # bootstrap 样本数量

bootstrap_weights_sp = []
for _ in range(n_bootstrap):
    # 使用原始数据进行 bootstrap 样本
    X_bootstrap,  y_bootstrap_sp, = resample(X, Y_sp,random_state=np.random.randint(1000))

    # 在 bootstrap 样本上拟合 PLSR 模型
    pls_bootstrap = PLSRegression(n_components=comp)

    pls_bootstrap.fit(X_bootstrap, y_bootstrap_sp)
    # bootstrap_weights_sp.append(pls_bootstrap.coef_.flatten())
    # print(len(pls_bootstrap.x_loadings_[:,0]))
    # bootstrap_weights_sp.append(pls_bootstrap.coef_[:,0])
    bootstrap_weights_sp.append(pls_bootstrap.x_loadings_[:, 0])
    # print(bootstrap_weights_sp[_])
    # bootstrap_weights_sp.append(pls_bootstrap.x_loadings_[:, 0])

# # 计算 bootstrap 标准误差

bootstrap_std_errors_sp = np.std(np.array(bootstrap_weights_sp), axis=0)
print("asdasd",bootstrap_std_errors_sp.shape)
# z_scores_sp = pls_weights_sp/ bootstrap_std_errors_sp
z_scores_sp = pls_weights_sp2/ bootstrap_std_errors_sp
print(z_scores_sp)
print(bootstrap_weights_sp)
p_values_sp = 2 * (1 - stats.norm.cdf(abs(z_scores_sp)))
# p_values_sp = [0 for i in range(18686)]
# # print(s)
# for i in range(18686):
#     for j in range(1000):
#         if bootstrap_weights_sp[j][i] >= z_scores_sp[i]:
#             p_values_sp[i]+=1
#
# for i in p_values_sp:
#     p_values_sp[i]/=1000

print(p_values_sp)
# # print("wqewqe",z_scores_sp)
# p_values_sp = 2 * (1 - norm.cdf(np.abs(z_scores_sp)))
# # print("p-val",p_values_sp)



# fdr_corrected_sp = multipletests(p_values_sp,  method='fdr_bh')[1]
fdr_corrected_sp = multipletests(p_values_sp,  method='fdr_bh')[1]
print("fdr_corrected_sp",fdr_corrected_sp)
significant_indices = np.where(fdr_corrected_sp < 0.05)[0]


# fdr_corrected_sort= np.argsort(fdr_corrected_sp)
# significant_variables_sp = sorted_indices_sp[:3000]
significant_variables_sp = []
# # 获取 FDR 校正后的 P 值小于 0.05 的自变量

# fdr_corrected_sort = np.where(fdr_corrected_sp < 0.05)[0]
# significant_variables_sp = []
#
# gg = [9353, 10507, 8482, 7852, 10371, 8633, 9723, 9037]
# gg =[1947, 6387, 2046, 2041, 1949, 10371,89839]
#
# gg1 = ["EFNB1" , "SLIT1", "EPHA8",  "EPHA1" ,"CXCL12" ,"EFNB3"]
# kk = 0
# weig = []
# # for ii in fdr_corrected_sort:
# for ii in significant_indices:
#     # if pls_weights_sp[ii]<0:
#         significant_variables_sp.append(sorted(gene_list_dir)[ii])
#         weig.append(pls_weights_sp[ii])
#         # kk +=1
#         if sorted(gene_list_dir)[ii] in gg:
#             print(sorted(gene_list_dir)[ii], pls_weights_sp[ii])
#         #
#         #
#         # if kk==3000:
#         #     break
#
# # print(weig)
# final_ = []
# aa = np.argsort(np.abs(weig))[::-1]
# for oo in aa:
#     final_.append(significant_variables_sp[oo])

import pandas as pd
data = {'gene_id': gene_list_dir,
        'weight': pls_weights_sp2,
        'z': z_scores_sp,
        'p':fdr_corrected_sp}

df = pd.DataFrame(data)

# 选择要保存的Excel文件名
excel_file = "G:\\IPMI_Jornal\\gene_data\\gene_data\\rs.xlsx"

# 将数据写入Excel文件
df.to_excel(excel_file, index=False, sheet_name='Sheet1', engine='openpyxl')


# np.savetxt("G:\\IPMI_Jornal\\gene_data\\gene_data\\BA28_sp_gene_sort_abs_final.txt",final_,fmt="%d")
# np.savetxt("G:\\IPMI_Jornal\\gene_data\\gene_data\\BA28_sp_gene_sort_abs.txt",final_,fmt="%d")
# np.savetxt("G:\\IPMI_Jornal\\gene_data\\gene_data\\charm_sp_gene.txt",significant_variables_sp,fmt="%d")
















































