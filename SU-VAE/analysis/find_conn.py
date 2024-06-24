"""2023 / 7 / 9
10: 36
dell"""

""""
图片平均值
"""
import os
import cv2
import numpy as np

# 设置文件夹路径和输出文件名
# folder_path = r"G:\IPMI_Jornal\chcp\eval\m_specific"
# output_filename = "G:\\IPMI_Jornal\\chcp\\eval\\average\\average_macaque_specific.jpg"
#
# # 获取所有图片文件的路径
# image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
#
# # 读取所有图片并计算平均值
# images = []
# for path in image_paths:
#     image = cv2.imread(path)
#     images.append(image)
#
# average_image = np.mean(images, axis=0).astype(np.uint8)
#
# # 输出平均值图像
# cv2.imwrite(output_filename, average_image)



# """
# 找到每个连接属于哪个独有、共有
# """
# #
# from PIL import Image
# import os
# import numpy as np
#
# xlabels_l3 = ['','Anterior cingulate cortex','Midcingulate cortex','Medial Orbital frontal cortex',
#               'Lateral Orbital frontal cortex','Caudal Orbital frontal cortex','FEF', 'Dorsolateral prefrontal cortex',
#               'Ventrolateral prefrontal cortex', 'Lateral Motor cortex','Medial supplementary Motor areas',
#               'Primary Somatosensory cortex','Second Somatosensory cortex','V6','Area 5','Ventromedial intraparietal sulcus',
#               'Lateral intraparietal sulcus','Medial superior temporal area','Area 7 in the inferior parietal lobule',
#               'Area 7 on the medial wall','Posterior cingulate gyrus','Parahippocampal cortex','Rhinal cortex','Temporal pole',
#               'Area TEO','Area TE','Fundus of the superior temporal sulcus','Rostral superior temporal region',
#               'Caudal superior temporal gyrus','belt areas of auditory cortex','core areas of auditory cortex',
#               'Floor of the lateral sulcus',
#            'Middle temporal area','visual areas 4','Preoccipital visual area 2-3','Primary visual cortex']
#
# image1 = Image.open('G:\\IPMI_Jornal\\input_all\\average_human.jpg').convert('L')
# image2 = Image.open('G:\\IPMI_Jornal\\chcp\\eval\\average\\average_human_shared.jpg').convert('L')
#
# image3 = Image.open('G:\\IPMI_Jornal\\chcp\\eval\\average\\average_human_specific.jpg').convert('L')
#
#
# image4 = Image.open('G:\\IPMI_Jornal\\input_all\\average_macaque.jpg').convert('L')
# # image5 = Image.open('G:\\IPMI_Jornal\\chcp\\eval\\average\\average_macaque_shared.jpg').convert('L')
# image6 = Image.open('G:\\IPMI_Jornal\\chcp\\eval\\average\\average_macaque_specific.jpg').convert('L')
#
#
# image = [image1, image2, image3,image4, image6, image6]
# array_sh = np.array(image[1])
# array_hsp = np.array(image[2])
# array_msp = np.array(image[5])
#
# d_sh = {}
# d_hsp = {}
# d_msp = {}
#
# k=1
# array = np.array(image[k])
#
#
# for y in range(array.shape[0]):
#     for x in range(array.shape[1]):
#         if  array_sh[y, x]>10:
#             sh = array_sh[y, x]
#         else:
#             sh = 0
#         if  array_hsp[y, x]>10:
#             hsp = array_hsp[y, x]
#         else:
#             hsp = 0
#         if  array_msp[y, x]>10:
#             msp = array_msp[y, x]
#         else:
#             msp = 0
#
#         if int(sh)-int(hsp)>5 and int(sh)-int(msp)>5 and x>y:
#             if y+1 in d_sh:
#                 d_sh[y+1].append(x+1)
#             else:
#                 d_sh[y + 1] = [x + 1]
#
#
#         elif int(hsp)-int(sh)>20 and int(hsp)-int(msp)>20 and x>y:
#             if y+1 in d_hsp:
#                 d_hsp[y+1].append(x+1)
#             else:
#                 d_hsp[y + 1] = [x + 1]
#
#         elif int(msp)-int(sh)>20 and int(msp)-int(hsp)>20 and x>y:
#             if y+1 in d_msp:
#                 d_msp[y+1].append(x+1)
#             else:
#                 d_msp[y + 1] = [x + 1]
#
# for i in range(35):
#     # if i+1 in d_hsp.keys():                # charm
#     #         print(str(i+1)+":",d_hsp[i+1])
#     #         print(xlabels_l3[i + 1] + ":", [xlabels_l3[k] for k in d_hsp[i + 1]])
#     if i+1 in d_msp.keys():
#             print(str(i+1)+":",d_msp[i+1])
#             # print(xlabels_l3[i+1] + ":", [xlabels_l3[k] for k in d_sh[i + 1]])


"""
找到连接属于独有、共有 (根据t-test)
"""
#
from PIL import Image
import os
import numpy as np


"""
BA28
"""
# human_specific_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\h_specific\\"
# macaque_specific_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\m_specific\\"
# shared_dir = "G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\h_shared\\"

"""
charm
"""
# human_specific_dir = "G:\\IPMI_Jornal\\CHARM_DATA\\eval\\h_specific\\all\\"
# macaque_specific_dir = "G:\\IPMI_Jornal\\CHARM_DATA\\eval\\m_specific\\"
# shared_dir = "G:\\IPMI_Jornal\\CHARM_DATA\\eval\\h_shared\\all\\"

"""
chcp
"""
human_specific_dir = "G:\\IPMI_Jornal\\chcp\\eval\\h_specific\\"
macaque_specific_dir = "G:\\IPMI_Jornal\\chcp\\eval\\m_specific\\"
shared_dir = "G:\\IPMI_Jornal\\chcp\\eval\\h_shared\\"


regions = 28  # BA 28  charm 35

# h_specific = [0 for i in range(378)]
# m_specific = [0 for i in range(378)]
# shared = [0 for i in range(378)]
"""BA28"""
# h_specific = [[] for _ in range(378)]
# m_specific = [[] for _ in range(378)]
# shared = [[] for _ in range(378)]
"""charm"""
# h_specific = [[] for _ in range(595)]
# m_specific = [[] for _ in range(595)]
# shared = [[] for _ in range(595)]

h_specific = [[] for _ in range(378)]
m_specific = [[] for _ in range(378)]
shared = [[] for _ in range(378)]

# print(h_specific)
# print(len(h_specific))
for i in os.listdir(human_specific_dir):
    k = 0
    p = human_specific_dir+i
    im = Image.open(p).convert('L')
    image = np.array(im)

    for m in range(0,regions):
        for n in range(0,m):
            # print(m,n)
            # print(k)
            # h_specific[k]+=image[m,n]/255
            h_specific[k].append(image[m, n] / 255)
            k+=1
len_h_sp = len(os.listdir(human_specific_dir))
# avg_h_specific = [h_specific[i]/len_h_sp for i in range(len(h_specific))]



for i in os.listdir(macaque_specific_dir):
    k = 0
    p = macaque_specific_dir+i
    im = Image.open(p).convert('L')
    image = np.array(im)

    for m in range(0,regions):
        for n in range(0,m):
            # print(m,n)
            # print(k)
            # m_specific[k]+=image[m,n]/255
            m_specific[k].append(image[m, n] / 255)
            k+=1
len_m_sp = len(os.listdir(macaque_specific_dir))
# avg_m_specific = [m_specific[i]/len_m_sp for i in range(len(m_specific))]

id_list = {}
idc=0
for m in range(0, regions):
    for n in range(0, m):
        id_list[idc] = (m+1,n+1)
        idc +=1
print(id_list)

for i in os.listdir(shared_dir):
    k = 0
    p = shared_dir+i
    im = Image.open(p).convert('L')
    image = np.array(im)

    for m in range(0,regions):
        for n in range(0,m):
            # print(m,n)
            # print(k)
            # shared[k]+=image[m,n]/255
            shared[k].append(image[m, n] / 255)
            k+=1
# len_sh = len(os.listdir(shared_dir))
# # avg_shared = [shared[i]/len_sh for i in range(len(shared))]
#
#
# print(len_sh,len_h_sp,len_m_sp)

# print(np.array(h_specific).shape)
# print(np.array(m_specific).shape)
# print(np.array(shared).shape)
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
rs_sh_p1 = []
rs_sh_p2 = []
rs_sh_p3 = []
rs_sh = []

rs_hsp = []
rs_hsp_p1 = []
rs_hsp_p2 = []
rs_hsp_p3 = []

rs_msp = []
rs_msp_p1 = []
rs_msp_p2 = []
rs_msp_p3 = []

sh_final = []
hsp_final = []
msp_final = []

for i in range(378): # charm 595
    s, p_value = stats.ttest_rel(shared[i][:885], h_specific[i][:885])
    t, p_value2 = stats.ttest_rel(shared[i][:885], m_specific[i])
    f_value, p_value3 = stats.f_oneway(shared[i][:885], h_specific[i][:885],m_specific[i])
    # print(s,t,p_value,p_value2,p_value3)

    # if s>10 and t>10 and p_value<0.05 and p_value2<0.05 and p_value3<0.05:
    if np.mean(shared[i][:885])>np.mean(h_specific[i][:885]) and np.mean(shared[i][:885])>np.mean(m_specific[i]) \
            and p_value < 0.05 and p_value2 < 0.05 and p_value3 < 0.05:
        rs_sh.append(i)
        rs_sh_p1.append(p_value)
        rs_sh_p2.append(p_value2)
        rs_sh_p3.append(p_value3)
print(len(rs_sh))
cor_p1 = multipletests(rs_sh_p1,  method='fdr_bh')[1]
cor_p2 = multipletests(rs_sh_p2,  method='fdr_bh')[1]
cor_p3 = multipletests(rs_sh_p3,  method='fdr_bh')[1]
# print(len(np.where(cor_p1<0.05)[0]))
# print(len(np.where(cor_p2<0.05)[0]))
# print(len(np.where(cor_p3<0.05)[0]))
# for i in rs_sh:
#     sh_final.append(id_list[i])
for i in range(len(rs_sh)):
    if cor_p1[i]<0.05 and cor_p2[i]<0.05 and cor_p3[i]<0.05:
        sh_final.append(id_list[rs_sh[i]])

for i in range(378):  #ba 378
    s, p_value = stats.ttest_rel(h_specific[i][:885],shared[i][:885],)
    t, p_value2 = stats.ttest_rel(h_specific[i][:885],m_specific[i])
    f_value, p_value3 = stats.f_oneway(shared[i][:885], h_specific[i][:885],m_specific[i])
    # print(s,t,p_value,p_value2,p_value3)

    if np.mean(h_specific[i][:885])>np.mean(shared[i][:885]) and np.mean(h_specific[i][:885]) > np.mean(m_specific[i]) \
            and p_value < 0.05 and p_value2 < 0.05 and p_value3 < 0.05:
        rs_hsp.append(i)
        rs_hsp_p1.append(p_value)
        rs_hsp_p2.append(p_value2)
        rs_hsp_p3.append(p_value3)
print(len(rs_hsp))
cor_p1 = multipletests(rs_hsp_p1,  method='fdr_bh')[1]
cor_p2 = multipletests(rs_hsp_p2,  method='fdr_bh')[1]
cor_p3 = multipletests(rs_hsp_p3,  method='fdr_bh')[1]
# print(len(np.where(cor_p1<0.05)[0]))
# print(len(np.where(cor_p2<0.05)[0]))
# print(len(np.where(cor_p3<0.05)[0]))
for i in range(len(rs_hsp)):
    if cor_p1[i]<0.05 and cor_p2[i]<0.05 and cor_p3[i]<0.05:
        hsp_final.append(id_list[rs_hsp[i]])

# print(len(hsp_final))


for i in range(378):
    s, p_value = stats.ttest_rel(m_specific[i],shared[i][:885],)
    t, p_value2 = stats.ttest_rel(m_specific[i],h_specific[i][:885])
    f_value, p_value3 = stats.f_oneway(shared[i][:885], h_specific[i][:885],m_specific[i])
    # print(s,t,p_value,p_value2,p_value3)

    if np.mean(m_specific[i]) > np.mean(shared[i][:885]) and np.mean(m_specific[i]) > np.mean(h_specific[i][:885]) \
            and p_value < 0.05 and p_value2 < 0.05 and p_value3 < 0.05:
        rs_msp.append(i)
        rs_msp_p1.append(p_value)
        rs_msp_p2.append(p_value2)
        rs_msp_p3.append(p_value3)
print(len(rs_msp))
cor_p1 = multipletests(rs_msp_p1,  method='fdr_bh')[1]
cor_p2 = multipletests(rs_msp_p2,  method='fdr_bh')[1]
cor_p3 = multipletests(rs_msp_p3,  method='fdr_bh')[1]
# print(len(np.where(cor_p1<0.05)[0]))
# print(len(np.where(cor_p2<0.05)[0]))
# print(len(np.where(cor_p3<0.05)[0]))
for i in range(len(rs_msp)):
    if cor_p1[i]<0.05 and cor_p2[i]<0.05 and cor_p3[i]<0.05:
        msp_final.append(id_list[rs_msp[i]])

# print(len(msp_final))

print(len(sh_final))
print(sh_final)
print(len(hsp_final))
print(hsp_final)
print(len(msp_final))
print(msp_final)

# print(np.mean(m_specific[0]))
# print(np.mean(h_specific[0][:885]))
# print(np.mean(shared[0][:885]))
# for i in msp_final:
#     print(i)

mat_sh = []
mat_hsp = []
mat_msp = []
for m in range(0, regions):
    o=[]
    for n in range(0, regions):
        if (m+1,n+1) in sh_final:
            o.append(1)
        else:
            o.append(0)
    mat_sh.append(o)

for m in range(0, regions):
    for n in range(0, regions):
        if mat_sh[m][n] == 1:
            mat_sh[n][m] =1



for m in range(0, regions):
    o=[]
    for n in range(0, regions):
        if (m+1,n+1)  in hsp_final:
            o.append(1)
        else:
            o.append(0)
    mat_hsp.append(o)

for m in range(0, regions):
    for n in range(0, regions):
        if mat_hsp[m][n] == 1:
            mat_hsp[n][m] =1

for m in range(0, regions):
    o=[]
    for n in range(0, regions):
        if (m+1,n+1)  in msp_final:
            o.append(1)
        else:
            o.append(0)
    mat_msp.append(o)

for m in range(0, regions):
    for n in range(0, regions):
        if mat_msp[m][n] == 1:
            mat_msp[n][m] =1
import scipy.io as scio
# cv2.imwrite("G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\sh.jpg", np.array(mat_sh))
# cv2.imwrite("G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\hsp.jpg", np.array(mat_hsp))
# cv2.imwrite("G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\msp.jpg", np.array(mat_msp))
# scio.savemat("G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\shared_ttest.mat",{'data': np.array(mat_sh)})
# scio.savemat("G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\human_specific_ttest.mat",{'data': np.array(mat_hsp)})
# scio.savemat("G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\macaque_specific_ttest.mat",{'data': np.array(mat_msp)})

"""
BA 28
"""
# origion_sh = scio.loadmat(r"G:\IPMI_Jornal\BRODMANN_DATA\28_transfer\eval\average3\average_shared.mat")['data']
# origion_msp = scio.loadmat(r"G:\IPMI_Jornal\BRODMANN_DATA\28_transfer\eval\average3\average_macaque_specific.mat")['data']
# origion_hsp = scio.loadmat(r"G:\IPMI_Jornal\BRODMANN_DATA\28_transfer\eval\average3\average_human_specific.mat")['data']

""""
charm
"""
# origion_sh = scio.loadmat(r"G:\IPMI_Jornal\CHARM_DATA\eval\average3\charm_average_shared.mat")['data']
# origion_msp = scio.loadmat(r"G:\IPMI_Jornal\CHARM_DATA\eval\average3\charm_average_macaque_specific.mat")['data']
# origion_hsp = scio.loadmat(r"G:\IPMI_Jornal\CHARM_DATA\eval\average3\charm_average_human_specific.mat")['data']

"""
CHCP
"""
# origion_sh = scio.loadmat(r"")['data']
# origion_msp = scio.loadmat(r"G:\IPMI_Jornal\CHARM_DATA\eval\average3\charm_average_macaque_specific.mat")['data']
# origion_hsp = scio.loadmat(r"G:\IPMI_Jornal\CHARM_DATA\eval\average3\charm_average_human_specific.mat")['data']

origion_sh = np.array(Image.open(r"G:\IPMI_Jornal\chcp\eval\average\average_human_shared.jpg").convert('L'))/255
origion_msp = np.array(Image.open(r"G:\IPMI_Jornal\chcp\eval\average\average_macaque_specific.jpg").convert('L'))/255
origion_hsp = np.array(Image.open(r"G:\IPMI_Jornal\chcp\eval\average\average_human_specific.jpg").convert('L'))/255


new_sh = origion_sh*mat_sh
new_msp = origion_msp*mat_msp
new_hsp = origion_hsp*mat_hsp
import matplotlib.pyplot as plt


plt.imshow(new_msp, cmap='GnBu')  # 选择合适的 colormap
plt.colorbar()  # 添加颜色条
# plt.savefig(r'G:\IPMI_Jornal\BRODMANN_DATA\28_transfer\eval\average3\new_hsp.jpg')  # 保存图像为文件
# scio.savemat("G:\\IPMI_Jornal\\BRODMANN_DATA\\28_transfer\\eval\\average3\\masked_human_specific_ttest.mat",{'data': new_hsp})

# plt.savefig(r'G:\IPMI_Jornal\CHARM_DATA\eval\average3\new_msp.jpg')  # 保存图像为文件
# scio.savemat("G:\\IPMI_Jornal\\CHARM_DATA\\eval\\average3\\masked_msp_ttest.mat",{'data': new_msp})

plt.savefig(r'G:\IPMI_Jornal\chcp\eval\average3\new_msp.jpg')  # 保存图像为文件
scio.savemat("G:\\IPMI_Jornal\\chcp\\eval\\average3\\masked_msp_ttest.mat",{'data': new_msp})