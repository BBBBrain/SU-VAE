"""2023 / 7 / 20
15: 18
dell
基因数据从网站表面163842（点）转移到32492点
"""
import os
import shutil
import zipfile
import nibabel as nib
import scipy.spatial as spt
import numpy as np
""""
第一步
解压基因数据到一个文件夹
"""





"""
删除文件夹
"""
# folder_path = "/media/disk10t/yl/mRNA_map_163842_unzip/"
#
# # 获取文件夹中所有文件的列表
# file_list = os.listdir(folder_path)
#
# # 遍历文件列表，逐个删除文件
# for file_name in file_list:
#     # 构建文件的完整路径
#     file_path = os.path.join(folder_path, file_name)
#
#     # 删除文件
#     shutil.rmtree(file_path)

""""
解压
"""
# counts = 0
# gene_dir = "/media/disk10t/yl/mRNA_map_163842/"
# unzip_dir = "/media/disk10t/yl/mRNA_map_163842_unzip/"
# gene_files = os.listdir(gene_dir)
# for gf in gene_files:
#     print(counts)
#     gf_path = gene_dir + gf
#     uz_path = unzip_dir + gf[:-4]
#     if not os.path.exists(uz_path):
#         os.makedirs(uz_path)
#     with zipfile.ZipFile(gf_path, 'r') as zip_ref:
#         # 解压ZIP文件到目标文件夹
#         zip_ref.extractall(uz_path)
#     counts += 1

""""
第2步
读取两个表面的球面数据和刚才解压的基因mgh文件，并配准得到32492表面的基因数据存储于/media/disk10t/yl/mRNA_map_reg_32492下的txt文件
"""



"""
读取163842球面信息 Lh
read points
"""
# tmp = []
# origin_points_L = []
# with open("/media/disk10t/yl/gene_data/temp_lh.sphere.vtk") as origin:
#     while 1:
#         line = origin.readline()
#         line = line.strip("\n").split(" ")
#         # print(line)
#         if "POINTS" in line:
#             points_num = int(line[1])
#             # print(points_num)
#             for i in range(points_num):
#                 line = origin.readline()
#                 # print(line)
#                 line = line.strip("\n").split(" ")
#                 for j in range(3):
#                     # print(line[j])
#                     tmp.append(float(line[j]))
#                 origin_points_L.append(tmp)
#                 tmp = []
#             break
#
# """
# 读取163842球面信息 Rh
# read points
# """
# tmp = []
# origin_points_R = []
# with open("/media/disk10t/yl/gene_data/temp_rh.sphere.vtk") as origin:
#     while 1:
#         line = origin.readline()
#         line = line.strip("\n").split(" ")
#         # print(line)
#         if "POINTS" in line:
#             points_num = int(line[1])
#             # print(points_num)
#             for i in range(points_num):
#                 line = origin.readline()
#                 # print(line)
#                 line = line.strip("\n").split(" ")
#                 for j in range(3):
#                     # print(line[j])
#                     tmp.append(float(line[j]))
#                 origin_points_R.append(tmp)
#                 tmp = []
#             break
#
#
#
#
# """
# 读取32492球面信息 LH
# read points
# """
# tmp = []
# new_points_L = []
# with open("/media/disk10t/yl/gene_data/lh.sphere_result1.reg.vtk") as origin:
#     while 1:
#         line = origin.readline()
#         line = line.strip("\n").split(" ")
#         # print(line)
#         if "POINTS" in line:
#             points_num = int(line[1])
#             # print(points_num)
#             for i in range(points_num):
#                 line = origin.readline()
#                 # print(line)
#                 line = line.strip("\n").split(" ")
#                 for j in range(3):
#                     # print(line[j])
#                     tmp.append(float(line[j]))
#                 new_points_L.append(tmp)
#                 tmp = []
#             break
# """
# 读取32492球面信息 RH
# read points
# """
# tmp = []
# new_points_R = []
# with open("/media/disk10t/yl/gene_data/rh.sphere_result1.reg.vtk") as origin:
#     while 1:
#         line = origin.readline()
#         line = line.strip("\n").split(" ")
#         # print(line)
#         if "POINTS" in line:
#             points_num = int(line[1])
#             # print(points_num)
#             for i in range(points_num):
#                 line = origin.readline()
#                 # print(line)
#                 line = line.strip("\n").split(" ")
#                 for j in range(3):
#                     # print(line[j])
#                     tmp.append(float(line[j]))
#                 new_points_R.append(tmp)
#                 tmp = []
#             break
#
# """
#   两个球面每个点找对应并存储到trans_arr
# """
# tree_L = spt.cKDTree(data=origin_points_L)
# tree_R = spt.cKDTree(data=origin_points_R)
# trans_arr_L = []
# trans_arr_R = []
# for pts in range(len(new_points_L)):
#       template_pts = new_points_L[pts]
#       _, indexs = tree_L.query(template_pts, k=1)
#       trans_arr_L.append(indexs)
#
# for pts2 in range(len(new_points_R)):
#       template_pts2 = new_points_R[pts2]
#       _, indexs2 = tree_R.query(template_pts2, k=1)
#       trans_arr_R.append(indexs2)
#
#
#
#
# gene_dir = "/media/disk10t/yl/mRNA_map_163842_unzip/"
# mgh_dir = os.listdir(gene_dir)
# for mf in mgh_dir:
#     new_L = []
#     new_R = []
#     r_mgh_path = gene_dir + mf + "/" + mf + "_mRNA_rh.mgh"
#     l_mgh_path = gene_dir + mf + "/" + mf + "_mRNA_lh.mgh"
#     r_mgh_img = nib.load(r_mgh_path)
#     l_mgh_img = nib.load(l_mgh_path)
#     l_mgh_data = l_mgh_img.get_fdata().reshape(-1)
#     r_mgh_data = r_mgh_img.get_fdata().reshape(-1)
#     for i in range(32492):
#         new_L.append(l_mgh_data[trans_arr_L[i]])
#         new_R.append(r_mgh_data[trans_arr_R[i]])
#
#     uz_path = "/media/disk10t/yl/mRNA_map_reg_32492/" + mf
#     if not os.path.exists(uz_path):
#         os.makedirs(uz_path)
#     np.savetxt(uz_path + "/" + mf +"_L.txt",new_L)
#     np.savetxt(uz_path + "/" + mf +"_R.txt",new_R)



""""
第3步
计算脑区间相关性的基因矩阵 
BA23数据 
"""
"""
读取脑区模板，循环读取基因数据，
计算每个脑区基因强度平均值，然后计算脑区间基因强度相关矩阵
存储矩阵
"""
import numpy as np
import scipy.io as scio
import os
import nibabel as nb
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

label_path = './Human.Brodmann09.32k_fs_LR.dlabel.nii'
img = nb.load(label_path)
label = img.get_fdata()[0]
part_L = label[:32492]
# print(part_L)
part_R = label[32492:]


# part_trans = {7:1,10:2,6:3,4:4,8:5,3:6,9:7,2:8,5:9,17:10,19:11,29:12,25:13,26:14,27:15,28:16,20:17,21:18,12:19,15:20,14:21,13:22,11:23,16:24,18:25,24:26,22:27,23:28}
part_trans = {54:8,55:6,56:4,57:9,58:3,59:1,60:5,61:7,62:2,63:31,64:40,65:44,66:45,67:23,68:39,69:43,70:19,71:47,72:41,73:30,74:22,75:42,76:21,77:38,78:37,79:20,80:32,81:24,82:10,83:25,84:11,85:46,86:17,
87:18,88:27,89:36,90:35,91:28,92:29,93:26,94:33}
needs = [54,55,56,57,58,59,60,61,62,67,70,74,76,79,81,82,83,84,86,87,88,91,93]
BAs = [1,2,3,4,5,6,7,8,9,10,11,17,18,19,20,21,22,23,24,25,26,27,28]
# BAs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

gene_files_dir = '/media/disk10t/yl/mRNA_map_reg_32492/'
gene_files = os.listdir(gene_files_dir)
k=1
for file in gene_files[:1]:
    sig_L = {}
    sig_R = {}
    sig_all = []
    sig_all_L = []
    sig_all_R = []
    gene_L_path = gene_files_dir + file + '/' + file + "_L.txt"
    gene_R_path = gene_files_dir + file + '/' + file + "_R.txt"
    gene_L = np.loadtxt(gene_L_path)
    gene_R = np.loadtxt(gene_R_path)

    # print(gene_L.shape)
    for i in range(32492):
        if int(part_L[i]) in needs:
           if part_trans[int(part_L[i])] in sig_L:

              sig_L[part_trans[int(part_L[i])]].append(gene_L[i])
           else:
              sig_L[part_trans[int(part_L[i])]] = [gene_L[i]]

        if int(part_R[i]) in needs:
           if part_trans[int(part_R[i])] in sig_R:
              sig_R[part_trans[int(part_R[i])]].append(gene_R[i])
           else:
              sig_R[part_trans[int(part_R[i])]] = [gene_R[i]]

    # for ii in BAs:
    #     sig_all_L.append([np.median(np.array(sig_L[ii])),0])
    # for ii in BAs:
    #     sig_all_R.append([np.median(np.array(sig_R[ii])),0])

    for ii in BAs:
        sig_all_L.append([np.mean(np.array(sig_L[ii]))])
    for ii in BAs:
        sig_all_R.append([np.mean(np.array(sig_R[ii]))])

    # print(sig_all_L)
    # print(sig_all_R)
    sig_all_L = np.array(sig_all_L)
    sig_all_R = np.array(sig_all_R)
    print(sig_all_L.shape)
    sig_all_L = pairwise_distances(sig_all_L.reshape(-1, 1))
    sig_all_L = 1 / (1 + sig_all_L)
    print(sig_all_L.shape)
    sig_all_R = pairwise_distances(sig_all_R.reshape(-1, 1))
    sig_all_R = 1 / (1 + sig_all_R)
    # print(sig_all_L)


    scio.savemat("/media/disk10t/yl/gene_data/BA23_gene_func_matrix/" + file + '_L.mat', mdict={'data': sig_all_L})
    scio.savemat("/media/disk10t/yl/gene_data/BA23_gene_func_matrix/" + file + '_R.mat', mdict={'data': sig_all_R})

    print(k)
    k+=1


