import jittor as jt
import jittor.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
import pandas as pd


def sampling(args):
    (z_mean, z_log_var) = args
    std = jt.exp((0.5 * z_log_var))
    eps = jt.randn_like(std)
    return eps*std + z_mean


def zeros_like(x):
    return jt.zeros_like(x)


class z_encoder_f(nn.Module):
    def __init__(self, out_channels=128,  intermediate_dim=128, latent_dim=2):
        super(z_encoder_f, self).__init__()
        self.s_conv1 = nn.Conv(3, 64, 2, 2, 0)  # no padding
        self.s_conv2 = nn.Conv(64, out_channels, 3, 2, 2)
        self.s_conv3 = nn.Conv(out_channels, out_channels, 3, 2, 2)
        self.bn = nn.BatchNorm(out_channels)
        self.max_pool = nn.Pool(2, 2)
        self.relu = nn.LeakyReLU()

        # generate latent vector Q(z|X)
        self.s_h_layer = nn.Linear(out_channels * 2 * 2, intermediate_dim)
        self.s_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.s_log_var_layer = nn.Linear(intermediate_dim, latent_dim)

        self.clas = nn.Linear(intermediate_dim, 2)
        self.sof = nn.Sigmoid()

    def execute(self, x):
        s_h = self.max_pool(self.relu(self.s_conv1(x)))
        s_h = self.max_pool(self.relu(self.s_conv2(s_h)))
        s_h = self.max_pool(self.relu(self.s_conv3(s_h)))
        # shape = s_h.shape
        # shape info needed to build decoder model
        # z_h = jt.flatten(z_h)
        s_h = jt.reshape(s_h, [s_h.shape[0], -1])
        s_h = self.s_h_layer(s_h)
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s



class s1_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(s1_encoder_f, self).__init__()
        self.s_conv1 = nn.Conv(3, 64, 2, 2, 0)  # no padding
        self.s_conv2 = nn.Conv(64, out_channels, 3, 2, 2)
        self.s_conv3 = nn.Conv(out_channels, out_channels, 3, 2, 2)
        self.bn = nn.BatchNorm(out_channels)
        self.max_pool = nn.Pool(2, 2)
        self.relu = nn.LeakyReLU()

        # generate latent vector Q(z|X)
        self.s_h_layer = nn.Linear(out_channels * 2 * 2, intermediate_dim)
        self.s_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.s_log_var_layer = nn.Linear(intermediate_dim, latent_dim)

        self.clas = nn.Linear(intermediate_dim, 2)
        self.sof = nn.Sigmoid()

    def execute(self, x):
        s_h = self.max_pool(self.relu(self.s_conv1(x)))
        s_h = self.max_pool(self.relu(self.s_conv2(s_h)))
        s_h = self.max_pool(self.relu(self.s_conv3(s_h)))
        # shape = s_h.shape
        # shape info needed to build decoder model
        # z_h = jt.flatten(z_h)
        s_h = jt.reshape(s_h, [s_h.shape[0], -1])
        s_h = self.s_h_layer(s_h)
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s





class decoder_f(nn.Module):
    def __init__(self, intermediate_dim=128, latent_dim=2):
        super(decoder_f, self).__init__()
        # build decoder model
        self.rel = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.Li1 = nn.Linear(latent_dim * 2, intermediate_dim)
        self.Li2 = nn.Linear(intermediate_dim, 128 * 2 * 2)
        self.d_conv1 = nn.ConvTranspose(128, 64, 3, stride=3, padding=1)  # 2-> 4
        self.d_conv2 = nn.ConvTranspose(64, 32, 3, stride=3, padding=1)  # 4->10
        self.d_conv3 = nn.ConvTranspose(32, 16, 3, stride=2, padding=1)  # 10->19
        self.d_conv4 = nn.ConvTranspose(16, 8, 2, stride=2, padding=2)  # 19->34
        self.d_conv5 = nn.ConvTranspose(8, 3, 3, stride=3, padding=1)  # 34->100

    def execute(self, latent_inputs):
        x = self.rel(self.Li1(latent_inputs))
        # print(x.shape)
        x = self.rel(self.Li2(x))
        # print(x.shape)
        # x = jt.reshape(x, [x.shape[0], 128, 2, 2])
        x = jt.reshape(x, [1, 128, 2, 2])
        # print(x.shape)
        x = self.rel(self.d_conv1(x))
        # print(x.shape)
        x = self.rel(self.d_conv2(x))
        x = self.rel(self.d_conv3(x))
        x = self.rel(self.d_conv4(x))

        # print(x.shape)
        x = self.sig(self.d_conv5(x))
        # print(x.shape)
        return x


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.z_encoder_f = z_encoder_f()
        self.s1_encoder_f = s1_encoder_f()

        self.decoder_f = decoder_f()
        self.dist = True
        self.dis = nn.Linear(2*2, 1)
        self.sig = nn.Sigmoid()
        # self.latent_dim = 16

    def execute(self, inputs1, inputs2, d):
        i1_z_mean, i1_z_log_var, i1_z = self.z_encoder_f(inputs1)  # 输入1的共享特征
        i2_z_mean, i2_z_log_var, i2_z = self.z_encoder_f(inputs2)  # 输入2的共享特征


        i1_s_mean, i1_s_log_var, i1_s = self.s1_encoder_f(inputs1)  # 输入1的独有特征
        zeros = zeros_like(i1_z)
        outputs1 = self.decoder_f(jt.concat([i1_s, i1_z], -1))  # 输入1重建
        outputs2 = self.decoder_f(jt.concat([zeros, i2_z], -1))  # 输入2重建
        out_s1 = self.decoder_f(jt.concat([i1_s, zeros], -1))  # 输入2重建
        out_z1 = self.decoder_f(jt.concat([zeros,i1_z], -1))  # 输入2重建

        # i1_s_output = self.decoder_f(jt.concat([i1_s, zeros], -1))  # 输入1独有特征重建
        # i2_s_output = self.decoder_f(jt.concat([zeros, i2_s], -1))  # 输入2独有特征重建
        # i1_z_output = self.decoder_f(jt.concat([zeros, i1_z], -1))  # 输入1共享特征重建
        # i2_z_output = self.decoder_f(jt.concat([zeros, i2_z], -1))  # 输入2共享特征重建

        q_bar_score = 0
        q_score = 0

        if d:
            z1 = i1_z[:20, :]
            z2 = i1_z[20:, :]

            s1 = i1_s[:20, :]
            s2 = i1_s[20:, :]

            q_bar = jt.concat(
                [jt.concat([s1, z2], 1),
                 jt.concat([s2, z1], 1)],
                0)

            q = jt.concat(
                [jt.concat([s1, z1], 1),
                 jt.concat([s2, z2], 1)],
                0)
            q_bar_score = (self.sig(self.dis(q_bar)) + .1) * .85  # +.1 * .85 so that it's 0<x<1
            q_score = (self.sig(self.dis(q)) + .1) * .85

        return [[i1_z_mean, i1_z_log_var, i1_z], [i2_z_mean, i2_z_log_var, i2_z], [i1_s_mean, i1_s_log_var, i1_s],
                [outputs1, outputs2],
                [q_score, q_bar_score],
                out_s1,
                out_z1
                ]
















# import jittor as jt
# import jittor.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score
# import os
# import pandas as pd
#
#
# def sampling(args):
#     (z_mean, z_log_var) = args
#     std = jt.exp((0.5 * z_log_var))
#     eps = jt.randn_like(std)
#     return eps*std + z_mean
#
#
# def zeros_like(x):
#     return jt.zeros_like(x)
#
#
# class z_encoder_f(nn.Module):
#     def __init__(self, out_channels=128,  intermediate_dim=128, latent_dim=16):
#         super(z_encoder_f, self).__init__()
#         self.relu = nn.LeakyReLU()
#         self.fc1 = nn.Linear(784 * 3, 256)
#         self.fc21 = nn.Linear(256, 16)
#         self.fc22 = nn.Linear(256, 16)
#
#     def execute(self, x):
#         x = jt.reshape(x, [x.shape[0], -1])
#         s_h = self.relu(self.fc1(x))
#
#         s_mean = self.fc21(s_h)
#         s_log_var = self.fc22(s_h)
#         s = sampling([s_mean, s_log_var])
#         return s_mean, s_log_var, s
#
#
#
# class s1_encoder_f(nn.Module):
#     def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=16):
#         super(s1_encoder_f, self).__init__()
#         self.relu = nn.LeakyReLU()
#         self.fc1 = nn.Linear(784 * 3, 256)
#         self.fc21 = nn.Linear(256, 16)
#         self.fc22 = nn.Linear(256, 16)
#
#     def execute(self, x):
#         x = jt.reshape(x, [x.shape[0], -1])
#         s_h = self.relu(self.fc1(x))
#
#         s_mean = self.fc21(s_h)
#         s_log_var = self.fc22(s_h)
#         s = sampling([s_mean, s_log_var])
#         return s_mean, s_log_var, s
#
#
#
#
#
# class decoder_f(nn.Module):
#     def __init__(self, intermediate_dim=128, latent_dim=16):
#         super(decoder_f, self).__init__()
#         # build decoder model
#         self.rel = nn.LeakyReLU()
#         self.sig = nn.Sigmoid()
#         self.Li1 = nn.Linear(latent_dim * 2, 256)
#         self.Li2 = nn.Linear(256, 784 * 3)
#         self.lab = nn.Linear(784 * 3, 3)
#
#     def execute(self, latent_inputs):
#         x = self.rel(self.Li1(latent_inputs))
#         # print(x.shape)
#         x = self.rel(self.Li2(x))
#
#         # print(x.shape)
#         x = jt.reshape(x, [x.shape[0], 3, 28, 28])
#
#         x = self.sig(x)
#         # print(x.shape)
#         return x
#
#
# class CVAE(nn.Module):
#     def __init__(self):
#         super(CVAE, self).__init__()
#         self.z_encoder_f = z_encoder_f()
#         self.s1_encoder_f = s1_encoder_f()
#
#         self.decoder_f = decoder_f()
#         self.dist = True
#         self.dis = nn.Linear(16*2, 1)
#         self.sig = nn.Sigmoid()
#         # self.latent_dim = 16
#
#     def execute(self, inputs1, inputs2, d):
#         i1_z_mean, i1_z_log_var, i1_z = self.z_encoder_f(inputs1)  # 输入1的共享特征
#         i2_z_mean, i2_z_log_var, i2_z = self.z_encoder_f(inputs2)  # 输入2的共享特征
#
#
#         i1_s_mean, i1_s_log_var, i1_s = self.s1_encoder_f(inputs1)  # 输入1的独有特征
#         zeros = zeros_like(i1_z)
#         outputs1 = self.decoder_f(jt.concat([i1_s, i1_z], -1))  # 输入1重建
#         outputs2 = self.decoder_f(jt.concat([zeros, i2_z], -1))  # 输入2重建
#         out_s1 = self.decoder_f(jt.concat([i1_s, zeros], -1))  # 输入2重建
#
#         # i1_s_output = self.decoder_f(jt.concat([i1_s, zeros], -1))  # 输入1独有特征重建
#         # i2_s_output = self.decoder_f(jt.concat([zeros, i2_s], -1))  # 输入2独有特征重建
#         # i1_z_output = self.decoder_f(jt.concat([zeros, i1_z], -1))  # 输入1共享特征重建
#         # i2_z_output = self.decoder_f(jt.concat([zeros, i2_z], -1))  # 输入2共享特征重建
#
#         q_bar_score = 0
#         q_score = 0
#
#         if d:
#             z1 = i1_z[:20, :]
#             z2 = i1_z[20:, :]
#
#             s1 = i1_s[:20, :]
#             s2 = i1_s[20:, :]
#
#             q_bar = jt.concat(
#                 [jt.concat([s1, z2], 1),
#                  jt.concat([s2, z1], 1)],
#                 0)
#
#             q = jt.concat(
#                 [jt.concat([s1, z1], 1),
#                  jt.concat([s2, z2], 1)],
#                 0)
#             q_bar_score = (self.sig(self.dis(q_bar)) + .1) * .85  # +.1 * .85 so that it's 0<x<1
#             q_score = (self.sig(self.dis(q)) + .1) * .85
#
#         return [[i1_z_mean, i1_z_log_var, i1_z], [i2_z_mean, i2_z_log_var, i2_z], [i1_s_mean, i1_s_log_var, i1_s],
#                 [outputs1, outputs2],
#                 [q_score, q_bar_score],
#                 out_s1
#                 ]