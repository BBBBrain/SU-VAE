import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

font = {'family': 'Times New Roman'}
plt.rc('font', **font)

# 示例数据
BA28_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\tsne\\"
BA23_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\eval\\tsne\\"
charm_dir = "G:\\IPMI_Jornal\\CHARM_DATA\\eval\\tsne\\"
chcp_dir = "G:\\IPMI_Jornal\\chcp\\eval\\tsne\\rs\\"
CVAE_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\CVAE\\cvae\\tsne\\"
DIG_dir = "D:\\Pycharm\\Projects\\kuawuzhong\\IPMI_Journal\\MedIA_revise\\infogan\\"  ##3  14
DID_dir  ="D:\\Pycharm\\Projects\\kuawuzhong\\IPMI_Journal\\MedIA_revise\\DID\\"  ##256
dir = DID_dir
k=122
data_group1 = np.transpose(pd.read_excel(dir + 'hms_zz1_' +  '0.xlsx')[[i for i in range(k)]].values).\
    reshape(k,256)[:,:256]  # 第一组数据，形状为 (100, 2)   shared
data_group2 = np.transpose(pd.read_excel(dir + 'hms_ss0_' +  '0.xlsx')[[i for i in range(k)]].values).\
    reshape(k,256)[:,:256]  # 第二组数据，形状为 (100, 2)   hs
data_group3 = np.transpose(pd.read_excel(dir + 'hms_ss1_' +  '0.xlsx')[[i for i in range(k)]].values).\
    reshape(k,256)[:,:256]  # 第三组数据，形状为 (100, 2)   ms
# data_group1 = np.transpose(pd.read_excel(dir + 'hms_zz1_' +  '0.xlsx')[[i for i in range(k)]].values).\
#     reshape(k,16)  # 第一组数据，形状为 (100, 2)   shared
# data_group2 = np.transpose(pd.read_excel(dir + 'hms_ss0_' +  '0.xlsx')[[i for i in range(k)]].values).\
#     reshape(k,16)  # 第二组数据，形状为 (100, 2)   hs

# 合并三组数据
data = np.concatenate((data_group1, data_group2, data_group3))
# data = np.concatenate((data_group1, data_group2))
# 创建 t-SNE 模型对象
tsne = TSNE(n_components=2, random_state=1000,n_iter=5000)

# 使用 t-SNE 对数据进行降维
embedded_data = tsne.fit_transform(data)
print(embedded_data.shape)
from sklearn.metrics import silhouette_score
pre1 = [0 for i in range(122)]  #3000  900(hm)
pre2 = [1 for i in range(122)]
pre3 = [2 for i in range(122)]
pre = pre1+pre2+pre3
# pre = pre1+pre2
a = []
b = []
c = []
for i in range(122):
    a.append(embedded_data[i].tolist())
for i in range(122,244):
    b.append(embedded_data[i].tolist())
for i in range(244,366):
    c.append(embedded_data[i].tolist())
datas = pd.DataFrame(a+b+c,columns=list('ab'))
# datas = pd.DataFrame(a+b,columns=list('ab'))
ss = round(silhouette_score(datas, pre), 3)
print(ss)
# # 创建标签
labels = np.repeat([1, 2, 3], k)  # 每个组的标签重复100次，分别为 1, 2, 3

# 创建颜色列表
colors = ['#FB5005', '#B4EEB4', '#00cdcd']
labelss=['Species-shared','Human-specific','Macaque-specific']
# 绘制降维后的数据
plt.figure(figsize=(12, 10))
for i in range(3):
    # plt.scatter(embedded_data[labels == (i+1), 0], embedded_data[labels == (i+1), 1], color=colors[i], label=f'Group {i+1}')
    plt.scatter(embedded_data[labels == (i + 1), 0], embedded_data[labels == (i + 1), 1], color=colors[i],
                label=labelss[i],s=70)

plt.title('t-SNE of latent features on BA data trained by DID',fontsize=30)
plt.xlabel('dimension 1',fontsize=29)
plt.ylabel('dimension 2',fontsize=29)
plt.xticks(fontsize=29)
plt.yticks(fontsize=29)
legend = plt.legend(fontsize=23,loc=(0.32, 0.75))
legend.legendHandles[0]._sizes = [140]  # 设置图标大小为40
legend.legendHandles[1]._sizes = [140]  # 设置图标大小为40
legend.legendHandles[2]._sizes = [140]  # 设置图标大小为40
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()
















# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import numpy as np
# import pandas as pd
#
# font = {'family': 'Times New Roman'}
# plt.rc('font', **font)
#
# # 示例数据
# BA28_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\tsne\\"
# BA23_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\eval\\tsne\\"
# charm_dir = "G:\\IPMI_Jornal\\CHARM_DATA\\eval\\tsne\\"
# chcp_dir = "G:\\IPMI_Jornal\\chcp\\eval\\tsne\\rs\\"
# CVAE_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\CVAE\\cvae\\RSA\\"
# dir = CVAE_dir
#
# k=122
#
# data_group1 = np.transpose(pd.read_excel(dir + 'hms_zz1_' +  '0.xlsx')[[i for i in range(k)]].values).\
#     reshape(k,16)  # 第一组数据，形状为 (100, 2)   shared
# data_group2 = np.transpose(pd.read_excel(dir + 'hms_ss0_' +  '0.xlsx')[[i for i in range(k)]].values).\
#     reshape(k,16)  # 第二组数据，形状为 (100, 2)   hs
#
#
# # 合并三组数据
# data = np.concatenate((data_group1, data_group2))
#
# # 创建 t-SNE 模型对象
# tsne = TSNE(n_components=2, random_state=1)
#
# # 使用 t-SNE 对数据进行降维
# embedded_data = tsne.fit_transform(data)
#
# # 创建标签
# labels = np.repeat([1, 2], k)  # 每个组的标签重复100次，分别为 1, 2, 3
#
# # 创建颜色列表
# colors = ['#FB5005', '#00cdcd']
# labelss=['Human-specific','Species-shared']
# # 绘制降维后的数据
# plt.figure(figsize=(12, 10))
# for i in range(2):
#     # plt.scatter(embedded_data[labels == (i+1), 0], embedded_data[labels == (i+1), 1], color=colors[i], label=f'Group {i+1}')
#     plt.scatter(embedded_data[labels == (i + 1), 0], embedded_data[labels == (i + 1), 1], color=colors[i],
#                 label=labelss[i],s=70)
#
# plt.title('t-SNE of latent features on BA28 data trained by CVAE',fontsize=30)
# plt.xlabel('dimension 1',fontsize=29)
# plt.ylabel('dimension 2',fontsize=29)
# plt.xticks(fontsize=29)
# plt.yticks(fontsize=29)
# # legend=plt.legend(fontsize=26,loc='upper right')
# legend = plt.legend(fontsize=23,loc=(0.72, 0.8))  # (0.85, 0.85) 表示图例位置的坐标参数
# legend.legendHandles[0]._sizes = [140]  # 设置图标大小为40
# legend.legendHandles[1]._sizes = [140]  # 设置图标大小为40
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.show()
