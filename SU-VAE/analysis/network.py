# """"
# 对图片进行graph分析
# """
# import os
# import random
#
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from scipy.stats import stats
# from skimage.segmentation import slic
# from skimage import io
#
# def calculate_robustness(graph):
#     original_size = len(graph)
#     largest_connected_component_size = max(len(comp) for comp in nx.connected_components(graph))
#     return largest_connected_component_size / original_size
#
# # 图像列表
# # dir0 = "G:\\IPMI_Jornal\\CHARM_DATA\\eval\\old\\h_shared\\"
# # dir1 = "G:\\IPMI_Jornal\\CHARM_DATA\\macaque\\mat\\"
# # dir2 = "G:\\IPMI_Jornal\\CHARM_DATA\\human\\his_mat\\L\\"
#
# # dir0 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\eval\\h_shared\\"
# # dir1 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\macaque\\mat\\"
# # dir2 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\human\\his_mat\\"
#
# dir0 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\h_shared\\"
# dir1 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\macaque\\mat\\"
# dir2 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\human\\his_mat\\"
#
# size = 28
#
# files0 = os.listdir(dir0)
# image_paths0  =[dir0+ f for f in files0]
# files1 = os.listdir(dir1)
# image_paths1  =[dir1+ f for f in files1]
# files2 = os.listdir(dir2)
# image_paths2  =[dir2+ f for f in files2]
# # 存储聚类系数结果
# clustering_coefficients0 = []
# clustering_coefficients1 = []
# clustering_coefficients2 = []
# clustering_coefficients3 = []
# import scipy.io as sio
# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# import networkx.algorithms.community as conn
# # 读取图像数据集，假设图像数据集存储在image_data列表中
# font = {'family': 'Times New Roman'}
# plt.rc('font', **font)
# # 存储每个图像的聚类系数
# cluster_coefficients = []
#
#
# "保留前10%的边"
# def keep_top_edges(G):
#     # 获取所有边的权重
#     edge_weights = nx.get_edge_attributes(G, 'weight')
#
#     # 按照权重值从大到小对边进行排序
#     sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
#
#     # 计算需要保留的边的数量
#     num_edges_to_keep = int(0.1 * len(sorted_edges))
#
#     # 保留前 num_edges_to_keep 个边，移除其他边
#     edges_to_keep = [edge for edge, weight in sorted_edges[:num_edges_to_keep]]
#
#     # 构建新的图，仅保留指定的边
#     new_G = G.edge_subgraph(edges_to_keep)
#
#     return new_G
#
# """"
# 随机生成
# """
# ss = [i for i in range(190,280)]
# print(ss)
# for i in range(477):
#     a = nx.generators.dense_gnm_random_graph(28,random.choice(ss))
#     # clustering_coefficient = nx.transitivity(a)
#     # clustering_coefficient = nx.average_degree_connectivity(a)
#     # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])
#     # clustering_coefficient = nx.local_efficiency(a)
#     # clustering_coefficient = nx.global_efficiency(a)
#     clustering_coefficient = nx.average_shortest_path_length(a)
#     clustering_coefficients3.append(clustering_coefficient)
#     # clustering_coefficient = nx.average_clustering(a)
#     # clustering_coefficients3.append(clustering_coefficient)
#     # clustering_coefficient = nx.average_shortest_path_length(a)
#     # c_len.append(clustering_coefficient)
#     # clustering_coefficient = nx.global_efficiency(a)
#     # c_ge.append(clustering_coefficient)
#     # clustering_coefficient = nx.local_efficiency(a)
#     # c_le.append(clustering_coefficient)
#     # clustering_coefficient = nx.average_degree_connectivity(a)
#     # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])
#     # c_conn.append(clustering_coefficient)
#     # clustering_coefficient = nx.transitivity(a)
#     # c_transi.append(clustering_coefficient)
#
#
#
#
#
#
#
#
#
#
# # 遍历图像数据集
# for ima in image_paths1[:800]:
#     mat_data = sio.loadmat(ima)
#
#     # 获取领接矩阵数据
#     adjacency_matrix = mat_data['data']
#
#     G = nx.Graph()
#     # avg = np.average(adjacency_matrix)
#     avg = np.sum(adjacency_matrix)
#     print("avg",avg)
#     num_nodes = adjacency_matrix.shape[0]
#     G.add_nodes_from(range(num_nodes))
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             weight = adjacency_matrix[i, j]
#             G.add_edge(i, j, weight=weight)
#             # if weight > 0.1*avg:
#             #         G.add_edge(i, j)
#             # else:
#             #         G.add_edge(i, j, weight=0)
#             # if weight > 0.02:  # 添加所有权重大于零的边
#             #     G.add_edge(i, j, weight=weight)
#     G = keep_top_edges(G)
#     # print(len(G.edges))
#
#     # avg = np.average(adjacency_matrix)
#     # for i in range(size):   ##修改大小
#     #     for j in range(size):
#     #         if adjacency_matrix[i][j]>avg:
#     #             adjacency_matrix[i][j] = 1
#     #         else:
#     #             adjacency_matrix[i][j] = 0
#     # G = nx.Graph(adjacency_matrix)
#     """
#     average_shortest_path_length
#     """
#
#     # subgraphs = list(nx.connected_components(G))
#     #
#     # # 对每个子图进行分析
#     # lens = 0
#     # for subgraph in subgraphs:
#     #     subgraph_diameter = nx.diameter(G.subgraph(subgraph))
#     #     subgraph_avg_shortest_path_length = nx.average_shortest_path_length(G.subgraph(subgraph))
#     #     lens +=subgraph_avg_shortest_path_length
#     # clustering_coefficients1.append(lens)
#     # # if lens < 2.3:
#     # #     clustering_coefficients1.append(lens)
#
#
#
#     # clustering_coefficient = nx.average_clustering(G)
#     # clustering_coefficient = nx.average_shortest_path_length(G)
#     try:
#         qq = 0
#         ii = 0
#         # clustering_coefficient = nx.average_shortest_path_length(G)
#         connected_subgraphs = list(nx.connected_components(G))
#         total_average_shortest_path = 0.0
#         # 计算每个连通子图的平均最短路径长度
#         for subgraph in connected_subgraphs:
#             subgraph_G = G.subgraph(subgraph)
#             average_shortest_path = nx.average_shortest_path_length(subgraph_G)
#             total_average_shortest_path += average_shortest_path
#         # 计算全局平均最短路径长度
#         clustering_coefficient = total_average_shortest_path / len(connected_subgraphs)
#
#         # for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
#         #     try:
#         #         qq += nx.average_shortest_path_length(C)
#         #         if qq>0:
#         #             ii += 1
#         #     except:
#         #         continue
#         # clustering_coefficient = qq / ii
#     except:
#         continue
#     # clustering_coefficient = nx.global_efficiency(G)
#     # clustering_coefficient = nx.degree_centrality(G)
#     # clustering_coefficient = nx.transitivity(G)
#     # clustering_coefficient = nx.average_node_connectivity(G)
#
#
#     # clustering_coefficient = nx.diameter(G)
#     # clustering_coefficient = nx.betweenness_centrality(G)  buxing
#     # clustering_coefficient = nx.local_efficiency(G)
#
#     # clustering_coefficient = nx.average_degree_connectivity(G)
#     # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])
#
#
#     # print(clustering_coefficient)
#     # if clustering_coefficient>0 and clustering_coefficient<1:
#     #     clustering_coefficients1.append(clustering_coefficient)
#     clustering_coefficients1.append(clustering_coefficient)
#
#
# for ima in image_paths2[:800]:
#     mat_data = sio.loadmat(ima)
#
#     # 获取领接矩阵数据
#     adjacency_matrix = mat_data['data']
#
#     G = nx.Graph()
#     # avg = np.average(adjacency_matrix)
#     avg = np.sum(adjacency_matrix)
#     num_nodes = adjacency_matrix.shape[0]
#     G.add_nodes_from(range(num_nodes))
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             weight = adjacency_matrix[i, j]
#             G.add_edge(i, j, weight=weight)
#             # if weight > 0.1*avg:
#             #         G.add_edge(i, j)
#             # else:
#             #         G.add_edge(i, j, weight=0)
#             # if weight > 0.02:  # 添加所有权重大于零的边
#             #     G.add_edge(i, j, weight=weight)
#     G = keep_top_edges(G)
#     # print(G.edges)
#     # clustering_coefficient = dict(nx.all_pairs_shortest_path_length(G))
#
#     # avg = np.average(adjacency_matrix)
#     # for i in range(size):
#     #     for j in range(size):
#     #         if adjacency_matrix[i][j] > avg:
#     #             adjacency_matrix[i][j] = 1
#     #         else:
#     #             adjacency_matrix[i][j] = 0
#     # print(adjacency_matrix)
#
#     # 创建Graph对象
#     # G = nx.Graph(adjacency_matrix)
#
#     """
#         average_shortest_path_length
#     """
#     # subgraphs = list(nx.connected_components(G))
#     #
#     # # 对每个子图进行分析
#     # lens = 0
#     # for subgraph in subgraphs:
#     #     subgraph_diameter = nx.diameter(G.subgraph(subgraph))
#     #     subgraph_avg_shortest_path_length = nx.average_shortest_path_length(G.subgraph(subgraph))
#     #     lens += subgraph_avg_shortest_path_length
#     # # if lens<2.3:
#     # #     clustering_coefficients2.append(lens)
#     # clustering_coefficients2.append(lens)
#
#
#     # clustering_coefficient = nx.average_clustering(G)
#     shortest_distances = dict(nx.shortest_path_length(G))
#
#     # 打印最短距离
#     for node, distances in shortest_distances.items():
#         print(f"从节点 {node} 到其他节点的最短距离：")
#         for target_node, distance in distances.items():
#             print(f"{target_node}: {distance}")
#     try:
#         qq = 0
#         ii = 0
#         # clustering_coefficient = nx.average_shortest_path_length(G)
#         connected_subgraphs = list(nx.connected_components(G))
#         print(connected_subgraphs)
#         total_average_shortest_path = 0.0
#         # 计算每个连通子图的平均最短路径长度
#         for subgraph in connected_subgraphs:
#             # a = dict(nx.all_pairs_shortest_path_length(subgraph))
#             # print(a)
#             subgraph_G = G.subgraph(subgraph)
#             average_shortest_path = nx.average_shortest_path_length(subgraph_G)
#             total_average_shortest_path += average_shortest_path
#         # 计算全局平均最短路径长度
#         clustering_coefficient = total_average_shortest_path / len(connected_subgraphs)
#
#         # for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
#         #     try:
#         #         qq += nx.average_shortest_path_length(C)
#         #         if qq > 0:
#         #             ii += 1
#         #     except:
#         #         continue
#         # print(qq,ii)
#         # clustering_coefficient = qq / ii
#     except:
#         continue
#
#     # clustering_coefficient = nx.global_efficiency(G)
#     # clustering_coefficient = nx.degree_centrality(G)
#     # clustering_coefficient = nx.transitivity(G)
#     # clustering_coefficient = nx.average_node_connectivity(G)
#
#     # clustering_coefficient = nx.diameter(G)
#     # clustering_coefficient = nx.betweenness_centrality(G)  buxing
#     # clustering_coefficient = nx.local_efficiency(G)
#
#     # clustering_coefficient = nx.average_degree_connectivity(G)
#     # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])
#
#     # print(clustering_coefficient)
#     # if clustering_coefficient > 0 and clustering_coefficient < 1:
#     #     clustering_coefficients2.append(clustering_coefficient)
#     clustering_coefficients2.append(clustering_coefficient)
#
# for ima in image_paths0[:800]:
#     adjacency_matrix = cv2.imread(ima)
#     adjacency_matrix = adjacency_matrix[:,:,0]/255
#
#     # print(adjacency_matrix)
#     adjacency_matrix = adjacency_matrix.reshape(size,size)
#
#     G = nx.Graph()
#     # avg = np.average(adjacency_matrix)
#     avg = np.sum(adjacency_matrix)
#
#     num_nodes = adjacency_matrix.shape[0]
#     print(num_nodes)
#     G.add_nodes_from(range(num_nodes))
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             weight = adjacency_matrix[i, j]
#             G.add_edge(i, j, weight=weight)
#             # if adjacency_matrix[i][j] > 0.1*avg:
#             #         G.add_edge(i, j)
#             # else:
#             #         G.add_edge(i, j, weight=weight)
#
#             # if weight > 0.02:  # 添加所有权重大于零的边
#             #     G.add_edge(i, j, weight=weight)
#     G = keep_top_edges(G)
#     # print(len(G.edges))
#     # avg = np.average(adjacency_matrix)
#     # for i in range(size):
#     #     for j in range(size):
#     #         if adjacency_matrix[i][j] > avg:
#     #             adjacency_matrix[i][j] = 1
#     #         else:
#     #             adjacency_matrix[i][j] = 0
#     # for i in range(35):
#     #     for j in range(35):
#     #         if adjacency_matrix[i][j]<5:
#     #             adjacency_matrix[i][j] = 0
#
#
#     # 创建Graph对象
#     # G = nx.Graph(adjacency_matrix)
#
#     """
#         average_shortest_path_length
#     """
#     # subgraphs = list(nx.connected_components(G))
#     # # 对每个子图进行分析
#     # lens = 0
#     # for subgraph in subgraphs:
#     #     subgraph_diameter = nx.diameter(G.subgraph(subgraph))
#     #     subgraph_avg_shortest_path_length = nx.average_shortest_path_length(G.subgraph(subgraph))
#     #     lens += subgraph_avg_shortest_path_length
#     # clustering_coefficients0.append(lens)
#     # # if lens<2.3:
#     # #     clustering_coefficients0.append(lens)
#
#     # clustering_coefficient = nx.average_clustering(G)
#     # clustering_coefficient = nx.average_shortest_path_length(G)
#
#     try:
#         qq = 0
#         ii = 0
#
#         # clustering_coefficient = nx.average_shortest_path_length(G)
#         connected_subgraphs = list(nx.connected_components(G))
#         total_average_shortest_path = 0.0
#         # 计算每个连通子图的平均最短路径长度
#         for subgraph in connected_subgraphs:
#             subgraph_G = G.subgraph(subgraph)
#             average_shortest_path = nx.average_shortest_path_length(subgraph_G)
#             total_average_shortest_path += average_shortest_path
#         # 计算全局平均最短路径长度
#         clustering_coefficient = total_average_shortest_path / len(connected_subgraphs)
#
#         # for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
#         #     try:
#         #         qq += nx.average_shortest_path_length(C)
#         #         if qq > 0:
#         #             ii += 1
#         #     except:
#         #         continue
#         # clustering_coefficient = qq / ii
#     except:
#         continue
#     # clustering_coefficient = nx.global_efficiency(G)
#     # clustering_coefficient = nx.degree_centrality(G)
#     # clustering_coefficient = nx.transitivity(G)
#     # clustering_coefficient = nx.average_node_connectivity(G)
#     # clustering_coefficient = nx.average_degree_connectivity(G)
#     # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])
#
#     # clustering_coefficient = nx.diameter(G)
#     # clustering_coefficient = nx.betweenness_centrality(G)  buxing
#     # clustering_coefficient = nx.local_efficiency(G)
#
#     # clustering_coefficient = nx.average_degree_connectivity(G)
#     # clustering_coefficient = np.average([v for n,v in clustering_coefficient.items()])
#
#     # print(clustering_coefficient)
#     clustering_coefficients0.append(clustering_coefficient)
#
# # print(cluster_coefficients)
# # 绘制箱线图
# box_colors = ['white', 'gray', '#B4EEB4', '#00cdcd']
# median_color = 'red'  # 设置中位数线的颜色
# data = [clustering_coefficients3, clustering_coefficients0[:477], clustering_coefficients1[:477], clustering_coefficients2[:477]]
# # print(len(data[0]),len(data[1]),len(data[2]))
# # print(data[0])
# # plt.boxplot(data)
# boxplot = plt.boxplot(data,labels=['Random','Shared','Macaque','Human'],patch_artist=True, medianprops=dict(color=median_color,linewidth=2),flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markersize': 6})
# for patch, color in zip(boxplot['boxes'], box_colors):
#     patch.set_facecolor(color)
# # print(data[0])
# # print(data[1])
# # print(data[2])
# # _,p_val_1_2 = stats.ttest_rel(data[0], data[1],equal_var=False)
# # _,p_val_1_2 = stats.ttest_ind(data[0], data[1],equal_var=False)
# print(len(data[0]))
# print(len(data[1]))
# print(len(data[2]))
# print(len(data[3]))
# t0,p_val_0_1 = stats.ttest_rel(data[0], data[1])
# t1,p_val_1_2 = stats.ttest_rel(data[1], data[2])
# t2,p_val_1_3 = stats.ttest_rel(data[1], data[3])
# t3,p_val_2_3 = stats.ttest_rel(data[2], data[3])
# # _,p_val_1_2 = stats.ttest_ind(data[0], data[1],equal_var=False)
# # _,p_val_1_3 = stats.ttest_ind(data[0], data[2],equal_var=False)
# # _,p_val_2_3 = stats.ttest_ind(data[1], data[2],equal_var=False)
# # print(t1,t2,t3)
# # print(p_val_1_2,p_val_1_3,p_val_2_3)
#
#
#
# """"
# charm average_shortest_path_length
# """
# # if p_val_1_2<0.0005:
# #     plt.axhline(y=2.05, xmin=0.2,xmax=0.5,color='black', linestyle='-')
# #     plt.text(1.5, 2.1, '***', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_1_2<0.005:
# #     plt.axhline(y=2.05, xmin=0.2,xmax=0.5,color='black', linestyle='-')
# #     plt.text(1.5, 2.1, '**', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_1_2<0.05:
# #     plt.axhline(y=2.05, xmin=0.2,xmax=0.5,color='black', linestyle='-')
# #     plt.text(1.5, 2.1, '*', color='red', fontsize=15, ha='center', va='center')
# #
# # if p_val_2_3 < 0.0005:
# #     plt.axhline(y=1.95, xmin=0.5, xmax=0.85, color='black', linestyle='-')
# #     plt.text(2.5, 2.0, '***', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_2_3 < 0.005:
# #     plt.axhline(y=1.95, xmin=0.5, xmax=0.85, color='black', linestyle='-')
# #     plt.text(2.5, 2.0, '**', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_2_3 < 0.05:
# #     plt.axhline(y=1.95, xmin=0.5, xmax=0.85, color='black', linestyle='-')
# #     plt.text(2.5, 2.0, '*', color='red', fontsize=15, ha='center', va='center')
#
# """"
# average_clustering
# """
# ymin, ymax = plt.ylim()
#
# # 修改y轴范围，使其包含横线的位置
# # plt.ylim(ymin, max(ymax, 2.3))  # charm average_shortest_path_length
#
# #glabal eff  1.05 0.97 1.01
# # avg sh pth  1.55
# # avg cluster coff 1.04 0.96 1.01   23/28 1.04 0.92 1.01
# #Average degree connectivity BA23  24 20 20.5 22.5 23   BA28 29.5 25 25.5 28 28.5  charm 36.5 32.5 33 35.5 36
# #Transitivity  0.95 1.02
#
# # plt.ylim(ymin, max(ymax, 1.02))
# liney_12 = 1.65
# texty_12  = 1.66
# liney_23 = 1.65
# texty_23  = 1.66
# # if p_val_1_2<0.0005:
# #     plt.axhline(y=liney_12, xmin=0.2,xmax=0.45,color='black', linestyle='-')
# #     plt.text(1.5, texty_12, '***', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_1_2<0.005:
# #     plt.axhline(y=liney_12, xmin=0.2,xmax=0.45,color='black', linestyle='-')
# #     plt.text(1.5, texty_12, '**', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_1_2<0.05:
# #     plt.axhline(y=liney_12, xmin=0.2,xmax=0.45,color='black', linestyle='-')
# #     plt.text(1.5, texty_12, '*', color='red', fontsize=15, ha='center', va='center')
# #
# # if p_val_2_3 < 0.0005:
# #     plt.axhline(y=liney_23, xmin=0.55, xmax=0.8, color='black', linestyle='-')
# #     plt.text(2.5, texty_23, '***', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_2_3 < 0.005:
# #     plt.axhline(y=liney_23, xmin=0.55, xmax=0.8, color='black', linestyle='-')
# #     plt.text(2.5, texty_23, '**', color='red', fontsize=15, ha='center', va='center')
# #
# # elif p_val_2_3 < 0.05:
# #     plt.axhline(y=liney_23, xmin=0.55, xmax=0.8, color='black', linestyle='-')
# #     plt.text(2.5, texty_23, '*', color='red', fontsize=15, ha='center', va='center')
#
#
# # 设置图表标题和坐标轴标签
# # plt.title('Average Shortest Path Length on CHARM Data',fontsize=16)
# # plt.title('Average shortest path length on BA28 data',fontsize=20)
# # plt.xlabel('Images')
# plt.ylabel('Transitivity',fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# # 显示图表
# plt.show()
# # print("shared-macaque:",p_val_1_2)
# # print("shared-human:",p_val_1_3)
# # print("macaque-human:",p_val_2_3)
# print(p_val_0_1)
# print(p_val_1_2)
# print(p_val_1_3)
# print(p_val_2_3)
# #Clustering Coefficient
# #Average degree connectivity
# #Local_Efficiency
# #Global Efficiency
# #Average Degree Centrality
# #Betweenness_Centrality
# # Transitivity





""""
对图片进行graph分析
"""
import os
import random

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import stats
from skimage.segmentation import slic
from skimage import io

def calculate_robustness(graph):
    original_size = len(graph)
    largest_connected_component_size = max(len(comp) for comp in nx.connected_components(graph))
    return largest_connected_component_size / original_size

# 图像列表
# dir0 = "G:\\IPMI_Jornal\\CHARM_DATA\\eval\\old\\h_shared\\"
# dir1 = "G:\\IPMI_Jornal\\CHARM_DATA\\macaque\\mat\\"
# dir2 = "G:\\IPMI_Jornal\\CHARM_DATA\\human\\his_mat\\L\\"

# dir0 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\eval\\h_shared\\"
# dir1 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\macaque\\mat\\"
# dir2 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\23_hu_ma\\human\\his_mat\\"

dir0 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\h_shared\\"
dir1 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\macaque\\mat\\"
dir2 = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\human\\his_mat\\"
dirsp = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\h_specific\\"
size = 28

files0 = os.listdir(dir0)
image_paths0  =[dir0+ f for f in files0]
files1 = os.listdir(dir1)
image_paths1  =[dir1+ f for f in files1]
files2 = os.listdir(dir2)
image_paths2  =[dir2+ f for f in files2]
# 存储聚类系数结果
clustering_coefficients0 = []
clustering_coefficients1 = []
clustering_coefficients2 = []
clustering_coefficients3 = []
clustering_coefficients4 = []
import scipy.io as sio
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import networkx.algorithms.community as conn
# 读取图像数据集，假设图像数据集存储在image_data列表中
font = {'family': 'Times New Roman'}
plt.rc('font', **font)
# 存储每个图像的聚类系数
cluster_coefficients = []


"保留前10%的边"
def keep_top_edges(G):
    # 获取所有边的权重
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # 按照权重值从大到小对边进行排序
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)

    # 计算需要保留的边的数量
    num_edges_to_keep = int(0.15* len(sorted_edges))

    # 保留前 num_edges_to_keep 个边，移除其他边
    edges_to_keep = [edge for edge, weight in sorted_edges[:num_edges_to_keep]]

    # 构建新的图，仅保留指定的边
    new_G = G.edge_subgraph(edges_to_keep)

    return new_G

""""
随机生成
"""
ss = [i for i in range(190,300)]
print(ss)
# random.seed(42)
for i in range(477):
    random.seed(i)
    print(random.choice(ss))
    a = nx.generators.dense_gnm_random_graph(28,random.choice(ss))


    # 生成带权重的随机图
    # n = 28  # 图中节点的数量
    # p = 0.95  # 边存在的概率
    min_weight = 0.001  # 最小权重值
    max_weight = 0.9 # 最大权重值
    # a = nx.gnp_random_graph(n, p)
    for (u, v) in a.edges():
        # 为每条边添加随机权重
        weight = random.uniform(min_weight, max_weight)
        print(weight)
        a[u][v]['weight'] = weight
        if weight>0.02:
            a[u][v]['weight'] = weight

    a = keep_top_edges(a)
    # print(a.edges)
    # clustering_coefficient = nx.transitivity(a)
    # clustering_coefficient = nx.average_degree_connectivity(a)
    # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])
    # clustering_coefficient = nx.local_efficiency(a)
    # clustering_coefficient = nx.global_efficiency(a)

    """
    shortest path len
    """
    # k = 0
    # c = 0
    # shortest_distances = dict(nx.shortest_path_length(a))
    # for node, distances in shortest_distances.items():
    #     # print(f"从节点 {node} 到其他节点的最短距离：")
    #     for target_node, distance in distances.items():
    #         # print(f"{target_node}: {distance}")
    #         if distance != 0:
    #             c += 1
    #             k += distance
    # clustering_coefficient = k / c
    """"
       degree
    """
    c = 0
    k = 0
    degrees = dict(a.degree())
    for node, distances in degrees.items():
        if distances != 0:
            c += 1
            k += distances
    clustering_coefficient = k / c

    # clustering_coefficient = nx.average_clustering(a)
    clustering_coefficients3.append(clustering_coefficient)
    # clustering_coefficient = nx.average_clustering(a)
    # clustering_coefficients3.append(clustering_coefficient)
    # clustering_coefficient = nx.average_shortest_path_length(a)
    # c_len.append(clustering_coefficient)
    # clustering_coefficient = nx.global_efficiency(a)
    # c_ge.append(clustering_coefficient)
    # clustering_coefficient = nx.local_efficiency(a)
    # c_le.append(clustering_coefficient)
    # clustering_coefficient = nx.average_degree_connectivity(a)
    # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])
    # c_conn.append(clustering_coefficient)
    # clustering_coefficient = nx.transitivity(a)
    # c_transi.append(clustering_coefficient)



def randomize_edges_with_weights(g):
    G = g.copy()
    for _ in range(20):
        # 随机选择两条边
        edge1, edge2 = random.sample(G.edges(), 2)

        # 交换它们的权重
        weight1 = G[edge1[0]][edge1[1]]['weight']
        weight2 = G[edge2[0]][edge2[1]]['weight']

        G[edge1[0]][edge1[1]]['weight'] = weight2
        G[edge2[0]][edge2[1]]['weight'] = weight1
    return G






# 遍历图像数据集
for ima in image_paths1[:800]:
    mat_data = sio.loadmat(ima)

    # 获取领接矩阵数据
    adjacency_matrix = mat_data['data']

    G = nx.Graph()
    # avg = np.average(adjacency_matrix)
    avg = np.sum(adjacency_matrix)
    # print("avg",avg)
    num_nodes = adjacency_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = adjacency_matrix[i, j]
            G.add_edge(i, j, weight=weight)
            # if weight > 0.1*avg:
            #         G.add_edge(i, j)
            # else:
            #         G.add_edge(i, j, weight=0)
            # if weight > 0.02:  # 添加所有权重大于零的边
            #     G.add_edge(i, j, weight=weight)

    G = keep_top_edges(G)
    print(len(G.edges))

    """
    average_shortest_path_length
    """
    # k=0
    # c=0
    # shortest_distances = dict(nx.shortest_path_length(G))
    # for node, distances in shortest_distances.items():
    #     print(f"从节点 {node} 到其他节点的最短距离：")
    #     for target_node, distance in distances.items():
    #         print(f"{target_node}: {distance}")
    #         if distance != 0:
    #             c += 1
    #             k += distance
    # clustering_coefficient = k / c

    """"
       degree
    """
    c = 0
    k = 0
    degrees = dict(G.degree())
    for node, distances in degrees.items():
        if distances != 0:
            c += 1
            k += distances
    clustering_coefficient = k / c

    # clustering_coefficient = nx.average_clustering(G)
    # clustering_coefficient = nx.global_efficiency(G)
    # clustering_coefficient = nx.degree_centrality(G)
    # clustering_coefficient = nx.transitivity(G)
    # clustering_coefficient = nx.average_node_connectivity(G)


    # clustering_coefficient = nx.diameter(G)
    # clustering_coefficient = nx.betweenness_centrality(G)  buxing
    # clustering_coefficient = nx.local_efficiency(G)

    # clustering_coefficient = nx.average_degree_connectivity(G)
    # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])


    # print(clustering_coefficient)
    # if clustering_coefficient>0 and clustering_coefficient<1:
    #     clustering_coefficients1.append(clustering_coefficient)
    clustering_coefficients1.append(clustering_coefficient)

# human
for ima in image_paths2[:800]:
    mat_data = sio.loadmat(ima)

    # 获取领接矩阵数据
    adjacency_matrix = mat_data['data']

    G = nx.Graph()
    # avg = np.average(adjacency_matrix)
    avg = np.sum(adjacency_matrix)
    num_nodes = adjacency_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = adjacency_matrix[i, j]
            G.add_edge(i, j, weight=weight)
            # if weight > 0.1*avg:
            #         G.add_edge(i, j)
            # else:
            #         G.add_edge(i, j, weight=0)
            # if weight > 0.02:  # 添加所有权重大于零的边
            #     G.add_edge(i, j, weight=weight)
    G = keep_top_edges(G)


    """
        average_shortest_path_length
    """
    # c=0
    # k=0
    # shortest_distances = dict(nx.shortest_path_length(G))
    # for node, distances in shortest_distances.items():
    #     print(f"从节点 {node} 到其他节点的最短距离：")
    #     for target_node, distance in distances.items():
    #         print(f"{target_node}: {distance}")
    #         if distance != 0:
    #             c += 1
    #             k += distance
    # clustering_coefficient = k / c
    """"
       degree
       """
    c = 0
    k = 0
    degrees = dict(G.degree())
    for node, distances in degrees.items():
        if distances != 0:
            c += 1
            k += distances
    clustering_coefficient = k / c



    clustering_coefficient = nx.global_efficiency(G)
    # clustering_coefficient = nx.degree_centrality(G)
    # clustering_coefficient = nx.transitivity(G)
    # clustering_coefficient = nx.average_node_connectivity(G)

    # clustering_coefficient = nx.diameter(G)
    # clustering_coefficient = nx.betweenness_centrality(G)  buxing
    # clustering_coefficient = nx.local_efficiency(G)

    # clustering_coefficient = nx.average_degree_connectivity(G)
    # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])

    # print(clustering_coefficient)
    # if clustering_coefficient > 0 and clustering_coefficient < 1:
    #     clustering_coefficients2.append(clustering_coefficient)
    clustering_coefficients2.append(clustering_coefficient)


#shared
pp=0
for ima in image_paths0[:800]:
    adjacency_matrix = cv2.imread(ima)
    adjacency_matrix = adjacency_matrix[:,:,0]/255
    adjacency_matrix = adjacency_matrix.reshape(size,size)

    G = nx.Graph()
    G_copy = nx.Graph()
    # avg = np.average(adjacency_matrix)
    # avg = np.sum(adjacency_matrix)
    # num_nodes = adjacency_matrix.shape[0]
    # print(num_nodes)
    G.add_nodes_from(range(28))
    G_copy.add_nodes_from(range(28))
    for i in range(28):
        for j in range(i + 1, 28):
            weight = adjacency_matrix[i, j]
            G_copy.add_edge(i, j, weight=weight)
            G.add_edge(i, j, weight=weight)
            # if adjacency_matrix[i][j] > 0.1*avg:
            #         G.add_edge(i, j)
            # else:
            #         G.add_edge(i, j, weight=weight)

            # if weight > 0.02:  # 添加所有权重大于零的边
            #     G.add_edge(i, j, weight=weight)

    G = keep_top_edges(G)


    # specific
    adjacency_matrix1 = cv2.imread(dirsp+"human_specific_"+files0[pp][-12:])
    adjacency_matrix1 = adjacency_matrix1[:, :, 0] / 255
    adjacency_matrix1 = adjacency_matrix1.reshape(size, size)

    G1 = nx.Graph()
    # avg = np.average(adjacency_matrix)

    # num_nodes1 = adjacency_matrix1.shape[0]
    # print(num_nodes)
    G1.add_nodes_from(range(28))
    for i in range(28):
        for j in range(i + 1, 28):
            weight = adjacency_matrix1[i, j]
            G1.add_edge(i, j, weight=weight)
            # if adjacency_matrix[i][j] > 0.1*avg:
            #         G.add_edge(i, j)
            # else:
            #         G.add_edge(i, j, weight=weight)

            # if weight > 0.02:  # 添加所有权重大于零的边
            #     G1.add_edge(i, j, weight=weight)

    print(G1.edges)
    G1 = randomize_edges_with_weights(G1.copy())


    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(range(28))
    # 添加graph1的边权重
    for u, v, weight in G_copy.edges(data='weight'):
        if combined_graph.has_edge(u, v):
            combined_graph[u][v]['weight'] += weight
        else:
            combined_graph.add_edge(u, v, weight=weight)
    # 添加graph2的边权重
    for u, v, weight in G1.edges(data='weight'):
        if combined_graph.has_edge(u, v):
            combined_graph[u][v]['weight'] += weight
        else:
            combined_graph.add_edge(u, v, weight=weight)
        # if combined_graph[u][v]['weight'] >0.9:
        #     combined_graph[u][v]['weight'] = 0.9
        # if combined_graph[u][v]['weight'] <0.02:
        #     combined_graph[u][v]['weight'] = 0
    print(combined_graph.edges(data='weight'))
    # combined_graph_1 = nx.Graph()
    # combined_graph_1.add_nodes_from(range(28))

    # for i in range(28):
    #     for j in range(i + 1, 28):
    #         weight = combined_graph[i][j]['weight']
    #         if weight > 0.03:  # 添加所有权重大于零的边
    #             combined_graph_1.add_edge(i, j, weight=weight)

    combined_graph = keep_top_edges(combined_graph)


    # print(len(G.edges))
    # avg = np.average(adjacency_matrix)
    # for i in range(size):
    #     for j in range(size):
    #         if adjacency_matrix[i][j] > avg:
    #             adjacency_matrix[i][j] = 1
    #         else:
    #             adjacency_matrix[i][j] = 0
    # for i in range(35):
    #     for j in range(35):
    #         if adjacency_matrix[i][j]<5:
    #             adjacency_matrix[i][j] = 0


    # 创建Graph对象
    # G = nx.Graph(adjacency_matrix)

    """
        average_shortest_path_length
    """
    # k=0
    # c=0
    # shortest_distances = dict(nx.shortest_path_length(G))
    # for node, distances in shortest_distances.items():
    #     print(f"从节点 {node} 到其他节点的最短距离：")
    #     for target_node, distance in distances.items():
    #         print(f"{target_node}: {distance}")
    #         if distance!=0:
    #             c+=1
    #             k+=distance
    # clustering_coefficient = k/c
    # """
    #   average_shortest_path_length  combined
    # """
    # k = 0
    # c = 0
    # shortest_distances = dict(nx.shortest_path_length(combined_graph))
    # for node, distances in shortest_distances.items():
    #     print(f"从节点 {node} 到其他节点的最短距离：")
    #     for target_node, distance in distances.items():
    #         print(f"{target_node}: {distance}")
    #         if distance != 0:
    #             c += 1
    #             k += distance
    # clustering_coefficient1 = k / c
    """"
    degree
    """
    c=0
    k=0
    degrees = dict(G.degree())
    for node, distances in degrees.items():
        if distances!=0:
            c+=1
            k+=distances
    clustering_coefficient = k / c
    """"
        degree combine
    """
    c = 0
    k = 0
    degrees = dict(combined_graph.degree())
    for node, distances in degrees.items():
        if distances != 0:
            c += 1
            k += distances
    clustering_coefficient1 = k / c

    # clustering_coefficient = nx.average_clustering(G)
    # clustering_coefficient1 = nx.average_clustering(combined_graph)
    # clustering_coefficient = nx.global_efficiency(G)
    # clustering_coefficient = nx.degree_centrality(G)
    # clustering_coefficient = nx.transitivity(G)
    # clustering_coefficient = nx.average_node_connectivity(G)
    # clustering_coefficient = nx.average_degree_connectivity(G)
    # clustering_coefficient = np.average([v for n, v in clustering_coefficient.items()])

    # clustering_coefficient = nx.diameter(G)
    # clustering_coefficient = nx.betweenness_centrality(G)  buxing
    # clustering_coefficient = nx.local_efficiency(G)

    # clustering_coefficient = nx.average_degree_connectivity(G)
    # clustering_coefficient = np.average([v for n,v in clustering_coefficient.items()])

    # print(clustering_coefficient)
    clustering_coefficients0.append(clustering_coefficient)
    clustering_coefficients4.append(clustering_coefficient1)
    pp+=1

# print(cluster_coefficients)
# 绘制箱线图
box_colors = ['white', 'gray', '#B4EEB4', '#00cdcd','white']
median_color = 'red'  # 设置中位数线的颜色
data = [clustering_coefficients3, clustering_coefficients0[:477], clustering_coefficients1[:477], clustering_coefficients2[:477],clustering_coefficients4[:477]]
# print(len(data[0]),len(data[1]),len(data[2]))
# print(data[0])
# plt.boxplot(data)
boxplot = plt.boxplot(data,labels=['Random','Shared','Macaque','Human',"Subs.Hu"],patch_artist=True, medianprops=dict(color=median_color,linewidth=2),flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markersize': 6})
for patch, color in zip(boxplot['boxes'], box_colors):
    patch.set_facecolor(color)

print(len(data[0]))
print(len(data[1]))
print(len(data[2]))
print(len(data[3]))
t0,p_val_0_1 = stats.ttest_rel(data[0], data[1])
t1,p_val_1_2 = stats.ttest_rel(data[1], data[2])
t2,p_val_1_3 = stats.ttest_rel(data[1], data[3])
t3,p_val_2_3 = stats.ttest_rel(data[2], data[3])


f_value, p_value = stats.f_oneway(data[0], data[1], data[2],data[3],data[4])

print("kkkkkkkkkk",f_value,p_value)

""""
average_clustering
"""
ymin, ymax = plt.ylim()

# 修改y轴范围，使其包含横线的位置
# plt.ylim(ymin, max(ymax, 2.3))  # charm average_shortest_path_length

#glabal eff  1.05 0.97 1.01
# avg sh pth  1.55
# avg cluster coff 1.04 0.96 1.01   23/28 1.04 0.92 1.01
#Average degree connectivity BA23  24 20 20.5 22.5 23   BA28 29.5 25 25.5 28 28.5  charm 36.5 32.5 33 35.5 36
#Transitivity  0.95 1.02

# plt.ylim(ymin, max(ymax, 1.02))
# nx.average_shortest_path_length()

plt.ylabel('Average degree',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# 显示图表
# plt.show()
# print("shared-macaque:",p_val_1_2)
# print("shared-human:",p_val_1_3)
# print("macaque-human:",p_val_2_3)
print(p_val_0_1)
print(p_val_1_2)
print(p_val_1_3)
print(p_val_2_3)
#        Average shortest path length
#Clustering coefficient
#Average degree connectivity
#Local_efficiency
#Global efficiency
#Average Degree Centrality
#Betweenness_Centrality
# Transitivity  Average shortest path length