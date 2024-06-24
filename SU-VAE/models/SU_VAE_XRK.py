import jittor as jt
import jittor.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
import random
np.random.seed(0)
random.seed(0)
def sampling(args):
    (z_mean, z_log_var) = args
    std = jt.exp((0.5 * z_log_var))
    eps = jt.randn_like(std)
    return eps*std + z_mean


def sampling_static(args):
    (z_mean, z_log_var) = args
    std = jt.exp((0.5 * z_log_var))
    return std + z_mean


def zeros_like(x):
    return jt.zeros_like(x)


class s2_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(s2_encoder_f, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab

class z2_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(z2_encoder_f, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab

class z1_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(z1_encoder_f, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab




class s2_encoder_f3(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(s2_encoder_f3, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


class s1_encoder_f3(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(s1_encoder_f3, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


class z_encoder_f3(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(z_encoder_f3, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab










class s2_encoder_f4(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=4):
        super(s2_encoder_f4, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


class s1_encoder_f4(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=4):
        super(s1_encoder_f4, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


class z_encoder_f4(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=4):
        super(z_encoder_f4, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab














# class z2_encoder_f3(nn.Module):
#     def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=16):
#         super(z2_encoder_f3, self).__init__()
#         self.clas = nn.Linear(400, 2)
#         self.sof = nn.Sigmoid()
#         self.relu = nn.LeakyReLU()
#         self.fc1 = nn.Linear(784*3, 400)
#         self.fc21 = nn.Linear(400, 16)
#         self.fc22 = nn.Linear(400, 16)
#
#     def execute(self, x):
#         x = jt.reshape(x, [x.shape[0], -1])
#         s_h = self.relu(self.fc1(x))
#         lab = self.sof(self.clas(s_h))
#         s_mean = self.fc21(s_h)
#         s_log_var = self.fc22(s_h)
#         s = sampling([s_mean, s_log_var])
#         return s_mean, s_log_var, s, lab


class zz1_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(zz1_encoder_f, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


class zz2_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(zz2_encoder_f, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


class xx1_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(xx1_encoder_f, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab


class xx2_encoder_f(nn.Module):
    def __init__(self, out_channels=128, intermediate_dim=128, latent_dim=2):
        super(xx2_encoder_f, self).__init__()
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
        lab = self.sof(self.clas(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        s = sampling([s_mean, s_log_var])
        return s_mean, s_log_var, s, lab




class decoder_f1(nn.Module):
    def __init__(self, intermediate_dim=256, latent_dim=2):
        super(decoder_f1, self).__init__()
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
        x = jt.reshape(x, [x.shape[0], 128, 2, 2])
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
# class decoder_f2(nn.Module):
#     def __init__(self, intermediate_dim=8, latent_dim=16):
#         super(decoder_f2, self).__init__()
#         # build decoder model
#         self.rel = nn.Relu()
#         self.sig = nn.Sigmoid()
#         self.Li1 = nn.Linear(latent_dim * 2, intermediate_dim)
#         self.Li2 = nn.Linear(intermediate_dim, 128 * 2 * 2)
#         self.lab = nn.Linear(128 * 2 * 2, 3)
#         self.d_conv1 = nn.ConvTranspose(128, 64, 3, stride=3, padding=1)  # 2-> 4
#         self.d_conv2 = nn.ConvTranspose(64, 32, 3, stride=3, padding=1)  # 4->10
#         self.d_conv3 = nn.ConvTranspose(32, 16, 3, stride=2, padding=1)  # 10->19
#         self.d_conv4 = nn.ConvTranspose(16, 8, 2, stride=2, padding=2)  # 19->34
#         self.d_conv5 = nn.ConvTranspose(8, 3, 3, stride=3, padding=1)  # 34->100
#
#     def execute(self, latent_inputs):
#         x = self.rel(self.Li1(latent_inputs))
#         # print(x.shape)
#         x = self.rel(self.Li2(x))
#
#         # print(x.shape)
#         x = jt.reshape(x, [x.shape[0], 128, 2, 2])
#         # print(x.shape)
#         x = self.rel(self.d_conv1(x))
#         # print(x.shape)
#         x = self.rel(self.d_conv2(x))
#         x = self.rel(self.d_conv3(x))
#         x = self.rel(self.d_conv4(x))
#
#         # print(x.shape)
#         x = self.sig(self.d_conv5(x))
#         # print(x.shape)
#         return x

class decoder_f3(nn.Module):
    def __init__(self, intermediate_dim=128, latent_dim=2):
        super(decoder_f3, self).__init__()
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
        x = jt.reshape(x, [x.shape[0], 128, 2, 2])
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

class decoder_f4(nn.Module):
    def __init__(self, intermediate_dim=128, latent_dim=4):
        super(decoder_f4, self).__init__()
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
        x = jt.reshape(x, [x.shape[0], 128, 2, 2])
        # x = jt.reshape(x, [1, 128, 2, 2])
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
class decoder_f(nn.Module):
    def __init__(self, intermediate_dim=256, latent_dim=2):
        super(decoder_f, self).__init__()
        self.rel = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.Li1 = nn.Linear(latent_dim * 2, intermediate_dim, bias=False)
        self.Li2 = nn.Linear(intermediate_dim, 128 * 2 * 2)
        self.lab = nn.Linear(128 * 2 * 2, 3)
        self.d_conv1 = nn.ConvTranspose(128, 64, 3, stride=3, padding=1)  # 2-> 4
        self.d_conv2 = nn.ConvTranspose(64, 32, 3, stride=3, padding=1)  # 4->10
        self.d_conv3 = nn.ConvTranspose(32, 16, 3, stride=2, padding=1)  # 10->19
        self.d_conv4 = nn.ConvTranspose(16, 8, 2, stride=2, padding=2)  # 19->34
        self.d_conv5 = nn.ConvTranspose(8, 3, 3, stride=3, padding=1)  # 34->100

    def execute(self, latent_inputs):
        x = self.rel(self.Li1(latent_inputs))
        # print(x.shape)
        x = self.rel(self.Li2(x))
        labs = self.sig(self.lab(x))
        # print(x.shape)
        x = jt.reshape(x, [x.shape[0], 128, 2, 2])
        # print(x.shape)
        x = self.rel(self.d_conv1(x))
        # print(x.shape)
        x = self.rel(self.d_conv2(x))
        x = self.rel(self.d_conv3(x))
        x = self.rel(self.d_conv4(x))

        # print(x.shape)
        x = self.sig(self.d_conv5(x))
        # print(x.shape)
        return x, labs


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.z1_encoder_f = z1_encoder_f()
        self.z2_encoder_f = z2_encoder_f()
        self.s1_encoder_f = s1_encoder_f()
        self.s2_encoder_f = s2_encoder_f()
        # self.xx1_encoder_f = xx1_encoder_f()
        # self.zz1_encoder_f = zz1_encoder_f()
        self.decoder_f = decoder_f()
        # self.decoder_f1 = decoder_f1()
        # self.decoder_f2 = decoder_f2()
        # self.decoder_f3 = decoder_f3()
        self.dis = nn.Linear(4, 1)
        self.dis2 = nn.Linear(4, 1)
        self.sig = nn.Sigmoid()




        # self.latent_dim = 16

    def execute(self, inputs1, inputs2, d, rate,rate2,zrate,zrate2):
        i1_z_mean, i1_z_log_var, i1_z, i1z_lab = self.z1_encoder_f(inputs1)
        i2_z_mean, i2_z_log_var, i2_z, i2z_lab = self.z2_encoder_f(inputs2)
        # i2_z_mean, i2_z_log_var, i2_z = self.z1_encoder_f(inputs2)

        i1_s_mean, i1_s_log_var, i1_s, i1s_lab = self.s1_encoder_f(inputs1)
        i2_s_mean, i2_s_log_var, i2_s, i2s_lab = self.s2_encoder_f(inputs2)
        # i2_s_mean, i2_s_log_var, i2_s = self.s1_encoder_f(inputs2)

        # i1_sd = i1_s.detach()
        # i2_sd = i2_s.detach()
        #
        # outputs1,_ = self.decoder_f(jt.concat([i1_sd, i2_z], -1))
        # outputs2,_ = self.decoder_f(jt.concat([i2_sd, i1_z], -1))

        outputs1, _ = self.decoder_f(jt.concat([1*i1_s, 1*i2_z], -1))
        outputs2, _ = self.decoder_f(jt.concat([1*i2_s, 1*i1_z], -1))

        zeros = zeros_like(i1_z)

        # i1_zz_mean, i1_zz_log_var, i1_zz, i1z_lab = self.zz1_encoder_f(inputs1)
        # i1_ss_mean, i1_ss_log_var, i1_ss, i1ss_lab = self.xx1_encoder_f(inputs1)
        # i1_ssd = i1_ss.detach()
        # outputs11 = self.decoder_f1(jt.concat([0.1*i1_ss, i1_zz], -1))
        # out_zz = self.decoder_f1(jt.concat([zeros, i1_zz], -1))
        # out_ss = self.decoder_f1(jt.concat([i1_ssd,zeros], -1))


        # _,i1labs = self.decoder_f(jt.concat([rate*i1_s, zeros], -1))
        # _,i2labs = self.decoder_f(jt.concat([rate*i2_s, zeros], -1))
        # _,i1labz = self.decoder_f(jt.concat([zeros, rate2*i1_z ], -1))
        # _,i2labz = self.decoder_f(jt.concat([zeros, rate2*i2_z ], -1))

        # i1_s_output_, _ = self.decoder_f(jt.concat([i1_sd, zeros], -1))
        # i2_s_output_, _ = self.decoder_f(jt.concat([i2_sd, zeros], -1))
        i1_s_output_, _ = self.decoder_f(jt.concat([i1_s, zeros], -1))
        i2_s_output_, _ = self.decoder_f(jt.concat([i2_s, zeros], -1))
        i1_z_output_, _ = self.decoder_f(jt.concat([zeros,  i1_z], -1))
        i2_z_output_, _ = self.decoder_f(jt.concat([zeros, i2_z], -1))

        # i1_zd = i1_z.detach()
        # i2_zd = i2_z.detach()
        # i1_sz_output,_ = self.decoder_f(jt.concat([i1_s, 1*i1_zd], -1))
        # i2_sz_output,_ = self.decoder_f(jt.concat([i2_s, 1*i2_zd], -1))
        i1_sz_output,_ = self.decoder_f(jt.concat([1*i1_s, i1_z], -1))
        i2_sz_output,_ = self.decoder_f(jt.concat([1*i2_s, i2_z], -1))

        q_bar_score = 0
        q_score = 0

        q_bar_score2 = 0
        q_score2 = 0

        if d:

            z1 = i1_z[:50, :]
            z2 = i1_z[50:, :]

            s1 = i1_s[:50, :]
            s2 = i1_s[50:, :]

            q_bar = jt.concat(
                [jt.concat([s1, z2], 1),
                 jt.concat([s2, z1], 1)],
                0)

            q = jt.concat(
                [jt.concat([s1, z1], 1),
                 jt.concat([s2, z2], 1)],
                0)
            q_bar_score = (self.sig(self.dis(q_bar)) + 0.1) *0.85  # +.1 * .85 so that it's 0<x<1
            q_score = (self.sig(self.dis(q)) + 0.1)*0.85
            # q_bar_score = self.sig(self.dis(q_bar))
            # q_score = self.sig(self.dis(q))

            z1_ = i2_z[:50, :]
            z2_ = i2_z[50:, :]

            s1_ = i2_s[:50, :]
            s2_ = i2_s[50:, :]

            q_bar2 = jt.concat(
                [jt.concat([s1_, z2_], 1),
                 jt.concat([s2_, z1_], 1)],
                0)

            q2 = jt.concat(
                [jt.concat([s1_, z1_], 1),
                 jt.concat([s2_, z2_], 1)],
                0)
            q_bar_score2 = (self.sig(self.dis2(q_bar2)) + 0.1)*0.85  # +.1 * .85 so that it's 0<x<1
            q_score2 = (self.sig(self.dis2(q2)) + 0.1) *0.85

        return [[i1_z_mean, i1_z_log_var, i1_z,i1z_lab], [i2_z_mean, i2_z_log_var, i2_z,i2z_lab], [i1_s_mean, i1_s_log_var, i1_s,i1s_lab],
                [i2_s_mean, i2_s_log_var, i2_s,i2s_lab],
                [i1_s_output_, i2_s_output_, i1_z_output_, i2_z_output_, outputs1, outputs2,i1_sz_output,i2_sz_output],
                [q_score, q_bar_score,q_score2, q_bar_score2]
                # [i1labs,i2labs,i1labz,i2labz]
                # [outputs11,out_zz,out_ss],
                # [i1_zz_mean, i1_zz_log_var, i1_zz],
                # [i1_ss_mean, i1_ss_log_var, i1_ss]
                ]


class CVAE2(nn.Module):
    def __init__(self):
        super(CVAE2, self).__init__()

        self.xx1_encoder_f = xx1_encoder_f()
        self.zz1_encoder_f = zz1_encoder_f()
        self.xx2_encoder_f = xx2_encoder_f()
        self.zz2_encoder_f = zz2_encoder_f()
        self.decoder_f1 = decoder_f1()



    def execute(self, inputs1, inputs2, d, rate, rate2):

        zz_mean, zz_log_var, zz, z_lab = self.zz2_encoder_f(inputs1)
        i1_ss_mean, i1_ss_log_var, i1_ss, i1ss_lab = self.xx1_encoder_f(inputs1)
        i2_ss_mean, i2_ss_log_var, i2_ss, i2ss_lab = self.xx2_encoder_f(inputs2)
        zeros = zeros_like(zz)
        # zzd = zz.detach()
        outputs11 = self.decoder_f1(jt.concat([1*i1_ss, 1*zz], -1))
        outputs22 = self.decoder_f1(jt.concat([1*i2_ss, 1*zz], -1))

        out_zz = self.decoder_f1(jt.concat([zeros, zz], -1))

        out_ss1 = self.decoder_f1(jt.concat([i1_ss, zeros], -1))
        out_ss2 = self.decoder_f1(jt.concat([i2_ss, zeros], -1))

        return [
                [outputs11, outputs22, out_zz, out_ss1,out_ss2],
                [zz_mean, zz_log_var, zz],
                [i1_ss_mean, i1_ss_log_var, i1_ss],
                [i2_ss_mean, i2_ss_log_var, i2_ss]
                ]


class CVAE3(nn.Module):
    def __init__(self):
        super(CVAE3, self).__init__()

        self.z_encoder_f = z_encoder_f3()
        self.s1_encoder_f = s1_encoder_f3()
        self.s2_encoder_f = s2_encoder_f3()
        self.decoder_f = decoder_f3()


    def execute(self, inputs1, inputs2, d, rate, rate2):
        i1_z_mean, i1_z_log_var, i1_z, i1z_lab = self.z_encoder_f(inputs1)
        i2_z_mean, i2_z_log_var, i2_z, i2z_lab = self.z_encoder_f(inputs2)

        i1_s_mean, i1_s_log_var, i1_s, i1s_lab = self.s1_encoder_f(inputs1)
        i2_s_mean, i2_s_log_var, i2_s, i2s_lab = self.s2_encoder_f(inputs2)

        outputs11 = self.decoder_f(jt.concat([1 * i1_s, 1 * i1_z], -1))
        outputs22 = self.decoder_f(jt.concat([1 * i2_s, 1 * i2_z], -1))

        zeros = zeros_like(i1_z)
        i1_s_output_ = self.decoder_f(jt.concat([i1_s, zeros], -1))
        i2_s_output_ = self.decoder_f(jt.concat([i2_s, zeros], -1))
        i1_z_output_ = self.decoder_f(jt.concat([zeros, i1_z], -1))
        i2_z_output_ = self.decoder_f(jt.concat([zeros, i2_z], -1))


        return [[i1_z_mean, i1_z_log_var, i1_z, i1z_lab], [i2_z_mean, i2_z_log_var, i2_z, i2z_lab],
                [i1_s_mean, i1_s_log_var, i1_s, i1s_lab],
                [i2_s_mean, i2_s_log_var, i2_s, i2s_lab],
                [i1_s_output_, i2_s_output_, i1_z_output_, i2_z_output_, outputs11, outputs22]
                ]


class CVAE4(nn.Module):
    def __init__(self):
        super(CVAE4, self).__init__()

        self.z_encoder_f = z_encoder_f4()
        self.s1_encoder_f = s1_encoder_f4()
        self.s2_encoder_f = s2_encoder_f4()
        self.decoder_f = decoder_f4()


    def execute(self, inputs1, inputs2, d, rate, rate2):
        i1_z_mean, i1_z_log_var, i1_z, i1z_lab = self.z_encoder_f(inputs1)
        i2_z_mean, i2_z_log_var, i2_z, i2z_lab = self.z_encoder_f(inputs2)

        i1_s_mean, i1_s_log_var, i1_s, i1s_lab = self.s1_encoder_f(inputs1)
        i2_s_mean, i2_s_log_var, i2_s, i2s_lab = self.s2_encoder_f(inputs2)

        outputs11 = self.decoder_f(jt.concat([1 * i1_s, 1 * i1_z], -1))
        outputs22 = self.decoder_f(jt.concat([1 * i2_s, 1 * i2_z], -1))

        zeros = zeros_like(i1_z)
        i1_s_output_ = self.decoder_f(jt.concat([i1_s, zeros], -1))
        i2_s_output_ = self.decoder_f(jt.concat([i2_s, zeros], -1))
        i1_z_output_ = self.decoder_f(jt.concat([zeros, i1_z], -1))
        i2_z_output_ = self.decoder_f(jt.concat([zeros, i2_z], -1))


        return [[i1_z_mean, i1_z_log_var, i1_z, i1z_lab], [i2_z_mean, i2_z_log_var, i2_z, i2z_lab],
                [i1_s_mean, i1_s_log_var, i1_s, i1s_lab],
                [i2_s_mean, i2_s_log_var, i2_s, i2s_lab],
                [i1_s_output_, i2_s_output_, i1_z_output_, i2_z_output_, outputs11, outputs22]
                ]