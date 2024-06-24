import os

import cv2
from jittor.dataset.dataset import Dataset
import jittor.transform as transform
import jittor as jt
import jittor.nn as nn
import numpy as np
from jittor_utils import LOG

# from SU_VAE2 import CVAE3   #  28
# from SU_VAE2 import CVAE4         ####23
from SU_VAE2_charm import CVAE4          ####23
from PIL import Image
import random
from jittor.optim import Adam
import pandas as pd
np.random.seed(0)
random.seed(0)

from jittor.dataset.dataset import ImageFolder

transform = transform.Compose([
    transform.ToTensor()
])

# val_i1_dir = './data_dependent/i1_eval'  独立数据集验证
# val_i2_dir = './data_dependent/i2_eval'
# val_i1_dir = './lunkuo_eval/i1_eval'
# val_i2_dir = './lunkuo_eval/i2_eval'
val_i1_dir = './data_charm/eval/RSA_data'  ##28
val_i2_dir = './data_charm/eval/RSA_data_m'
# val_i1_dir = './data_BA23/eval/RSA_data'
# val_i2_dir = './data_BA23/eval/RSA_data_m'
# val_i1_dir = './data_charm/eval/RSA_data'
# val_i2_dir = './data_charm/eval/RSA_data_m'
val_loader1 = ImageFolder(val_i1_dir).set_attrs(transform=transform, batch_size=1, shuffle=False)
val_loader2 = ImageFolder(val_i2_dir).set_attrs(transform=transform, batch_size=1, shuffle=False)

h_dir = val_loader1.imgs
m_dir = val_loader2.imgs

# fs =  os.listdir(val_i2_dir+"/1")
# print(fs)

jt.flags.use_cuda = 1
model = CVAE4()

times = 10

def val(model, val_loader1, val_loader2):
    model.eval()
    for ti in range(10):
        ss = {}
        zz = {}
        ss1 = {}
        ss2 = {}
        ss3 = {}
        ss4 = {}
        for d1,d2 in zip(enumerate(val_loader1), enumerate(val_loader2)):

            idx = d1[0]
            if ti ==0:
              print(h_dir[idx][0][:-6])
              # print(d1)
            # name = d1[1][2]
            # print(name)
            # print(val_loader1)
            input1 = d1[1][0]
            input2 = d2[1][0]
            # name = d2[1][2]
            # print(name)
            # print(input2)
            outputs = model(input1, input2,False, 0.01, 1)  # 0.1 0.5 1 1
            # zz_mean, zz_log_var, i_zz = out[1][0], out[1][1], out[1][2]
            # i1_s_mean, i1_s_log_var, i1_s = out[2][0], out[2][1], out[2][2]
            # outputs11, outputs22, out_zz, out_ss1, out_ss2 =  out[0][0],out[0][1],out[0][2],out[0][3],out[0][4]

            i1_z_mean, i1_z_log_var, i1_z = outputs[0][0], outputs[0][1], outputs[0][2]

            i2_z_mean, i2_z_log_var, i2_z = outputs[1][0], outputs[1][1], outputs[1][2]
            i1_s_mean, i1_s_log_var, i1_s = outputs[2][0], outputs[2][1], outputs[2][2]
            i2_s_mean, i2_s_log_var, i2_s = outputs[3][0], outputs[3][1], outputs[3][2]

            outputs11, outputs22, out_zz, out_z2, out_ss1, out_ss2 = outputs[4][4], outputs[4][5], outputs[4][2], \
                                                                     outputs[4][3], \
                                                                     outputs[4][0], outputs[4][1]

            # im1 = (out_zz[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/"+str(times)+"/shared/" + "human_shared_" + str(idx) + ".jpg", im1)
            # im_i1_s = (out_ss1[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/"+str(times)+"/specific/" + "human_specific_" + str(idx) + ".jpg", im_i1_s)
            # im_i2_s = (out_ss2[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/"+str(times)+"/m_specific/" + "monkey_specific_" + str(idx) + ".jpg", im_i2_s)
            # im_ss = (out_z2[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/"+str(times)+"/m_shared/" + "monkey_shared_" + str(idx) + ".jpg", im_ss)

            # im1 = (out_zz[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/" + "/shared/" + "human_shared_" + str(idx) + ".jpg", im1)
            # im_i1_s = (out_ss1[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/"  + "/specific/" + "human_specific_" + str(idx) + ".jpg",
            #             im_i1_s)
            # im_i2_s = (out_ss2[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/"  + "/m_specific/" + "monkey_specific_" + str(idx) + ".jpg",
            #             im_i2_s)
            # im_ss = (out_z2[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data/recon_eval/"  + "/m_shared/" + "monkey_shared_" + str(idx) + ".jpg", im_ss)

            """
            独立数据集
            """
            # im_i2 = (outputs22[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data_dependent/monkey_recon/" + "monkey_" + str(idx) + ".jpg", im_i2)
            #
            # im_i2_s = (out_ss2[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data_dependent/m_specific/" + "monkey_specific_" + str(idx) + ".jpg", im_i2_s)
            #
            # im_ss = (out_z2[0].numpy().transpose(1, 2, 0)) * 255
            # cv2.imwrite("./data_dependent/m_shared/" + "monkey_shared_" + str(idx) + ".jpg", im_ss)
            # ss1[idx] = i1_s_mean[0]
            # ss2[idx] = i2_s_mean[0]
            # ss3[idx] = i1_z_mean[0]
            # ss4[idx] = i2_z_mean[0]




            """aaaaaaaaa"""
            ss1[idx] = i1_s[0]
            ss2[idx] = i2_s[0]
            ss3[idx] = i1_z[0]
            ss4[idx] = i2_z[0]
        df1 = pd.DataFrame(ss1)
        df2 = pd.DataFrame(ss2)
        df3 = pd.DataFrame(ss3)
        df4 = pd.DataFrame(ss4)



        #     ss[idx] = i1_s[0]
        #     zz[idx] = i1_z[0]
        # df1 = pd.DataFrame(ss)
        # df2 = pd.DataFrame(zz)






        """aaaaaaaaa"""
        with pd.ExcelWriter('./data_charm/eval/RSA/hms_ss0_'+ str(ti)+'.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Sheet1')
        with pd.ExcelWriter('./data_charm/eval/RSA/hms_ss1_'+str(ti)+'.xlsx') as writer:
            df2.to_excel(writer, sheet_name='Sheet1')
        with pd.ExcelWriter('./data_charm/eval/RSA/hms_zz1_'+str(ti)+'.xlsx') as writer:
            df3.to_excel(writer, sheet_name='Sheet1')
        with pd.ExcelWriter('./data_charm/eval/RSA/hms_zz2_' + str(ti) + '.xlsx') as writer:
            df4.to_excel(writer, sheet_name='Sheet1')







        # with pd.ExcelWriter('./mu_sigma_v3/ss_'+str(ti)+'.xlsx') as writer:
        #     df1.to_excel(writer, sheet_name='Sheet1')
        # with pd.ExcelWriter('./mu_sigma_v3/zz_'+str(ti)+'.xlsx') as writer:
        #     df2.to_excel(writer, sheet_name='Sheet1')
        # # with pd.ExcelWriter('./mu_sigma/SU_for_monkey/ss_final_'+str(times)+'.xlsx') as writer:
        # #     df1.to_excel(writer, sheet_name='Sheet1')
        # # with pd.ExcelWriter('./mu_sigma/SU_for_monkey/zz_final_'+str(times)+'.xlsx') as writer:
        # #     df2.to_excel(writer, sheet_name='Sheet1')


    # with open("./mu_sigma/12_z.txt", "w") as f:
    #     for i in ss:
    #             f.write(str(i)+"\n")




print("loading........")
# model.load("./checkpoints/SU2/3200SU_final_data_net3.pkl")
# model.load("./checkpoints/BA_28/3300net4.pkl")
# model.load("./checkpoints/BA_23/3250net4.pkl")
model.load("./checkpoints/charm/3450net4.pkl")
with jt.no_grad():

    val(model,val_loader1,val_loader2)
