import os

import cv2
from jittor.dataset.dataset import ImageFolder
import jittor.transform as transform
import jittor as jt
import jittor.nn as nn
import numpy as np
from SU_VAE_XRK import CVAE
from SU_VAE_XRK import CVAE2
from SU_VAE_XRK import CVAE3
from SU_VAE_XRK import CVAE4
from PIL import Image
import random
from jittor.optim import Adam
import scipy.stats
from scipy.spatial.distance import pdist
np.random.seed(0)
random.seed(0)
# batch_size = 10
batch_size = 60
transform1 = transform.Compose([

    # transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transform.ToTensor()
    #   transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3])
    # transform.ToTensor()
])
transform2 = transform.Compose([
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.6, 0.6, 0.6], std=[1, 1, 1])
    #   transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3])
    # transform.ToTensor()
])
train_i1_dir = './data_back/xrk2/i1_train'
train_i2_dir = './data_back/xrk2/i2_train'
train_loader1 = ImageFolder(train_i1_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
train_loader2 = ImageFolder(train_i2_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
val_i1_dir = './data_back/xrk2/i1_eval'
val_i2_dir = './data_back/xrk2/i2_eval'
val_loader1 = ImageFolder(val_i1_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)
val_loader2 = ImageFolder(val_i2_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)


train_i3_dir = './data_back/xrk2/i1_train'
train_i4_dir = './data_back/xrk2/i2_train'
train_loader3 = ImageFolder(train_i3_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
train_loader4 = ImageFolder(train_i4_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
val_i3_dir = './data_back/xrk2/i1_eval'
val_i4_dir = './data_back/xrk2/i2_eval'
val_loader3 = ImageFolder(val_i3_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)
val_loader4 = ImageFolder(val_i4_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)

train_i5_dir = './data_back/xrk2/i1_train'
train_i6_dir = './data_back/xrk2/i2_train'
train_loader5 = ImageFolder(train_i5_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
train_loader6 = ImageFolder(train_i6_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
val_i5_dir = './data_back/xrk2/i1_eval'
val_i6_dir = './data_back/xrk2/i2_eval'
val_loader5 = ImageFolder(val_i5_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)
val_loader6 = ImageFolder(val_i6_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)



def loss(out, inputs1, inputs2, uni_lab,uni_lab2,shared_lab,epoch, beta=1, con_gamma=20000, zz_gamma=50000, sz_gamma=0):
    i1_z_mean, i1_z_log_var, i1_z,i1z_lab = out[0][0], out[0][1], out[0][2],out[0][3]
    i2_z_mean, i2_z_log_var, i2_z,i2z_lab = out[1][0], out[1][1], out[1][2],out[1][3]

    i1_s_mean, i1_s_log_var, i1_s,i1s_lab = out[2][0], out[2][1], out[2][2],out[2][3]
    i2_s_mean, i2_s_log_var, i2_s,i2s_lab = out[3][0], out[3][1], out[3][2],out[3][3]
    # i1labs, i2labs, i1labz, i2labz = out[6][0], out[6][1], out[6][2],out[6][3]
    # print("i1_s_mean", i1_s_mean)
    # print( "i1_s_log_var", i1_s_log_var)
    # print( "i1_s",i1_s)

    # print(uni_lab,shared_lab)
    # if epoch >5 and epoch%5==0:
    #     print("i1_z",i1_z)
    #     print("i1_s",i1_s)
    i1_s_output, i2_s_output, i1_z_output, i2_z_output, outputs1, outputs2, q_score, q_bar_score,q_score2, q_bar_score2,i1_sz_output,i2_sz_output = out[4][0], out[4][1], out[4][2], out[4][3], out[4][4], out[4][5], out[5][0], out[5][1],out[5][2], out[5][3], out[4][6], out[4][7]

    tc_loss = jt.log(q_score / (1 - q_score))
    discriminator_loss = - jt.log(q_score) - jt.log(1 - q_bar_score)
    tc_loss2 = jt.log(q_score2 / (1 - q_score2))
    discriminator_loss2 = - jt.log(q_score2) - jt.log(1 - q_bar_score2)
    # print(tc_loss,tc_loss2,discriminator_loss2,discriminator_loss)
    # print("tc",tc_loss,discriminator_loss)
    # reconstruction_loss = nn.mse_loss(jt.flatten(outputs1), jt.flatten(inputs1))*(20000 + epoch/500*50)
    # reconstruction_loss += nn.mse_loss(jt.flatten(outputs2), jt.flatten(inputs2))*(20000 + epoch/500*50)
    # if epoch<800:
    #     reconstruction_loss = nn.mse_loss(jt.flatten(outputs1), jt.flatten(inputs1)) * 20000
    #     reconstruction_loss += nn.mse_loss(jt.flatten(outputs2), jt.flatten(inputs2)) * 20000
    #     reconstruction_loss += nn.mse_loss(jt.flatten(i1_z_output), jt.flatten(i2_z_output)) * 8000
    # else:
    # print(shared_lab.shape, i1s_lab.shape)
    # print(outputs1.shape, inputs1.shape)

    reconstruction_loss = nn.mse_loss(jt.flatten(outputs1), jt.flatten(inputs1)) * 20000
    reconstruction_loss += nn.mse_loss(jt.flatten(outputs2), jt.flatten(inputs2)) * 20000
    reconstruction_loss += nn.mse_loss(jt.flatten(i1_sz_output), jt.flatten(inputs1)) * 10000
    reconstruction_loss += nn.mse_loss(jt.flatten(i2_sz_output), jt.flatten(inputs2)) * 10000
    reconstruction_loss += nn.mse_loss(jt.flatten(i1_z_output), jt.flatten(i2_z_output)) * 5000
    # reconstruction_loss += nn.mse_loss(jt.flatten(i2_sz_output), jt.flatten(inputs2)) * 10000
    # if epoch>10:
    # reconstruction_loss += nn.mse_loss(jt.flatten(i1_s_output), jt.flatten(jt.clamp(inputs1-i1_z_output,0,1))) * 7000
    # reconstruction_loss += nn.mse_loss(jt.flatten(i2_s_output), jt.flatten(jt.clamp(inputs2-i2_z_output,0,1))) * 7000
    # zzz1 = i1_z_output.numpy()
    # zzz2 = i2_z_output.numpy()
    # # print(jt.flatten(jt.array(np.triu(zzz1))))
    # # print(jt.flatten(jt.array(np.tril(zzz1))))
    # reconstruction_loss += nn.mse_loss(jt.flatten(jt.array(np.triu(zzz1))), jt.flatten(jt.array(np.tril(zzz1)))) * 1500
    # reconstruction_loss += nn.mse_loss(jt.flatten(jt.array(np.triu(zzz2))), jt.flatten(jt.array(np.tril(zzz2)))) * 1500
    # reconstruction_loss += coss(jt.flatten(i1_z_output), jt.flatten(i1_s_output)) * 2000
    # reconstruction_loss += coss(jt.flatten(i2_z_output), jt.flatten(i2_s_output)) * 2000


    # lab_loss = nn.bce_loss(uni_lab, i1labs) * 64*10000/10
    # lab_loss += nn.bce_loss(uni_lab2, i2labs) * 64*10000/10
    # lab_loss += nn.bce_loss(shared_lab, i1labz) * 64*10000/10
    # lab_loss += nn.bce_loss(shared_lab, i2labz) * 64 * 10000/10
    # reconstruction_loss += nn.cross_entropy_loss(shared_lab, i1z_lab) * 30000
    # reconstruction_loss += nn.cross_entropy_loss(shared_lab, i2z_lab) * 30000





    kl_loss = 1 + i1_z_log_var - jt.pow(i1_z_mean,2) - jt.exp(i1_z_log_var)

    kl_loss += (1 + i1_s_log_var - jt.pow(i1_s_mean, 2) - jt.exp(i1_s_log_var))

    kl_loss += 1 + i2_z_log_var - jt.pow(i2_z_mean,2) - jt.exp(i2_z_log_var)

    kl_loss +=( 1 + i2_s_log_var - jt.pow(i2_s_mean,2) - jt.exp(i2_s_log_var))




    # kl_loss += (1 + i1_z_log_var - jt.pow(i1_z_mean, 2) - jt.exp(i1_z_log_var)) + (1 + i2_z_log_var - jt.pow(i2_z_mean,2) - jt.exp(i2_z_log_var))
    # kl_loss += (1 + i_z_log_var - jt.pow(i_z_mean, 2) - jt.exp(i_z_log_var))

    kl_loss = kl_loss.sum(dim=-1)
    kl_loss *= -0.5
    # print("kl_loss", kl_loss)
    # cvae_loss = jt.mean(reconstruction_loss + beta * kl_loss + 200*(tc_loss + tc_loss2 ) + discriminator_loss + discriminator_loss2)
    cvae_loss = jt.mean(
        reconstruction_loss + kl_loss)
    # cvae_loss = jt.mean(reconstruction_loss + beta * kl_loss )
    # cvae_loss = jt.mean(reconstruction_loss + beta * kl_loss )
    return cvae_loss


def loss2(out, inputs1, inputs2,z1_out):
    # i1_z_mean, i1_z_log_var, i1_z = out[2][0], out[2][1], out[2][2]
    # i1_zz_mean, i1_zz_log_var, i1_zz = out[4][0], out[4][1], out[4][2]
    zz_mean, zz_log_var, zz = out[1][0], out[1][1], out[1][2]
    i1_s_mean, i1_s_log_var, i1_s = out[2][0], out[2][1], out[2][2]
    i2_s_mean, i2_s_log_var, i2_s = out[3][0], out[3][1], out[3][2]
    outputs11, outputs22, out_zz, out_ss1, out_ss2 = out[0][0], out[0][1], out[0][2], out[0][3], out[0][4]

    reconstruction_loss = nn.mse_loss(jt.flatten(outputs11), jt.flatten(inputs1)) * 15000
    reconstruction_loss += nn.mse_loss(jt.flatten(outputs22), jt.flatten(inputs2)) * 15000
    reconstruction_loss += nn.mse_loss(jt.flatten(out_ss1), jt.flatten(jt.clamp(inputs1-z1_out,0,1))) * 10000
    reconstruction_loss += nn.mse_loss(jt.flatten(out_ss2), jt.flatten(jt.clamp(inputs2-z1_out,0,1))) * 10000
    # print(jt.clamp(inputs1-z1_out,0,1)*255)
    # print(jt.clamp(inputs2 - z1_out, 0, 1) * 255)
    reconstruction_loss += nn.mse_loss(jt.flatten(out_zz), jt.flatten(z1_out)) * 10000



    kl_loss = 1 + zz_log_var - jt.pow(zz_mean, 2) - jt.exp(zz_log_var)
    # kl_loss += 1 + i1_z_log_var - jt.pow(i1_z_mean, 2) - jt.exp(i1_z_log_var)
    kl_loss += (1 + i1_s_log_var - jt.pow(i1_s_mean, 2) - jt.exp(i1_s_log_var))
    kl_loss += (1 + i2_s_log_var - jt.pow(i2_s_mean, 2) - jt.exp(i2_s_log_var))

    kl_loss = kl_loss.sum(dim=-1)
    kl_loss *= -0.5

    cvae_loss = jt.mean(reconstruction_loss + kl_loss )
    return cvae_loss


def loss4(out, inputs1, inputs2,z1_out,z2_out):

    i1_z_mean, i1_z_log_var, i1_z = out[0][0], out[0][1], out[0][2]
    i2_z_mean, i2_z_log_var, i2_z = out[1][0], out[1][1], out[1][2]
    i1_s_mean, i1_s_log_var, i1_s = out[2][0], out[2][1], out[2][2]
    i2_s_mean, i2_s_log_var, i2_s = out[3][0], out[3][1], out[3][2]

    i1_s_output_, i2_s_output_, i1_z_output_, i2_z_output_, outputs1, outputs2 = out[4][0], out[4][1], out[4][2], \
                                                                                 out[4][3], out[4][4], out[4][5]

    reconstruction_loss = nn.mse_loss(jt.flatten(outputs1), jt.flatten(inputs1)) * 15000
    reconstruction_loss += nn.mse_loss(jt.flatten(outputs2), jt.flatten(inputs2)) * 15000
    reconstruction_loss += nn.mse_loss(jt.flatten(i1_s_output_), jt.flatten(jt.clamp(inputs1-z1_out,0,1))) * 10000
    reconstruction_loss += nn.mse_loss(jt.flatten(i2_s_output_), jt.flatten(jt.clamp(inputs2-z2_out,0,1))) * 10000
    # print(jt.clamp(inputs1-z1_out,0,1)*255)
    # print(jt.clamp(inputs2 - z1_out, 0, 1) * 255)
    reconstruction_loss += nn.mse_loss(jt.flatten(i1_z_output_), jt.flatten(z1_out)) * 10000
    reconstruction_loss += nn.mse_loss(jt.flatten(i2_z_output_), jt.flatten(z2_out)) * 10000

    kl_loss = 1 + i1_z_log_var - jt.pow(i1_z_mean, 2) - jt.exp(i1_z_log_var)

    kl_loss += (1 + i1_s_log_var - jt.pow(i1_s_mean, 2) - jt.exp(i1_s_log_var))

    kl_loss += 1 + i2_z_log_var - jt.pow(i2_z_mean, 2) - jt.exp(i2_z_log_var)

    kl_loss += (1 + i2_s_log_var - jt.pow(i2_s_mean, 2) - jt.exp(i2_s_log_var))

    kl_loss = kl_loss.sum(dim=-1)
    kl_loss *= -0.5

    cvae_loss = jt.mean(reconstruction_loss + kl_loss )
    return cvae_loss

def loss3(out, inputs1,inputs2,  s1_out, s2_out):
    i1_z_mean, i1_z_log_var, i1_z = out[0][0], out[0][1], out[0][2]

    i2_z_mean, i2_z_log_var, i2_z = out[1][0], out[1][1], out[1][2]
    i1_s_mean, i1_s_log_var, i1_s = out[2][0], out[2][1], out[2][2]
    i2_s_mean, i2_s_log_var, i2_s = out[3][0], out[3][1], out[3][2]

    i1_s_output_, i2_s_output_, i1_z_output_, i2_z_output_, outputs1, outputs2 = out[4][0], out[4][1], out[4][2], out[4][3], out[4][4], out[4][5]




    reconstruction_loss = nn.mse_loss(jt.flatten(outputs2), jt.flatten(inputs2))* 15000
    reconstruction_loss += nn.mse_loss(jt.flatten(outputs1), jt.flatten(inputs1)) * 15000

    reconstruction_loss += nn.mse_loss(jt.flatten(i1_s_output_), jt.flatten(s1_out)) * 15000
    reconstruction_loss += nn.mse_loss(jt.flatten(i2_s_output_), jt.flatten(s2_out)) * 15000

    # reconstruction_loss += nn.mse_loss(jt.flatten(i1_z_output_), jt.flatten(i2_z_output_)) * 10000

    tmz1 = jt.clamp(inputs1 - s1_out, 0, 1)
    tmz2 = jt.clamp(inputs2 - s2_out, 0, 1)

    zzzz = jt.array(np.minimum(tmz1.numpy(),tmz2.numpy()))
    reconstruction_loss += nn.mse_loss(jt.flatten(i1_z_output_), jt.flatten(zzzz) )* 10000
    reconstruction_loss += nn.mse_loss(jt.flatten(i2_z_output_), jt.flatten(zzzz)) * 10000
    # reconstruction_loss += nn.mse_loss(jt.flatten(i1_z_output_), jt.flatten(jt.clamp(inputs1 - s1_out, 0, 1))) * 10000
    # reconstruction_loss += nn.mse_loss(jt.flatten(i2_z_output_), jt.flatten(jt.clamp(inputs2 - s2_out, 0, 1))) * 10000


    kl_loss = 1 + i1_z_log_var - jt.pow(i1_z_mean, 2) - jt.exp(i1_z_log_var)

    kl_loss += (1 + i1_s_log_var - jt.pow(i1_s_mean, 2) - jt.exp(i1_s_log_var))

    kl_loss += 1 + i2_z_log_var - jt.pow(i2_z_mean, 2) - jt.exp(i2_z_log_var)

    kl_loss += (1 + i2_s_log_var - jt.pow(i2_s_mean, 2) - jt.exp(i2_s_log_var))

    kl_loss = kl_loss.sum(dim=-1)
    kl_loss *= -0.5

    cvae_loss = jt.mean(reconstruction_loss + kl_loss)
    return cvae_loss


jt.flags.use_cuda = 1
lr = 0.001
lr2 = 0.001
train_epoch = 10000
img_size = 100
momentum = 0.9
weight_decay = 1e-5
model = CVAE()
model2 = CVAE2()
model3 = CVAE3()
model4 = CVAE4()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = Adam(model2.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = Adam(model3.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = Adam(model4.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer_s1 = Adam(model.s1_encoder_f.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer_s2 = Adam(model.s2_encoder_f.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer_z1 = Adam(model.z1_encoder_f.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer_z2 = Adam(model.z2_encoder_f.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer_d= Adam(model.decoder_f.parameters(), lr=lr, weight_decay=weight_decay)





def train(model, train_loader1, train_loader2, lossfun, optimizer, epoch,rate,rate2):
    model.train()
    train_loss = []
    for d1,d2 in zip(enumerate(train_loader1), enumerate(train_loader2)):
        idx = d1[0]
        input1 = d1[1][0]
        tar_shared = jt.array([0, 0,1])

        input2 = d2[1][0]
        tar_unique1 = jt.array([0, 1,0])

        tar_unique2 = jt.array([1,0, 0])

        outputs = model(input1, input2, False, 1, 1, 0.5, 1)  # 0.1 0.5 1 1

        loss = lossfun(outputs, input1, input2, tar_unique1,tar_unique2,tar_shared,epoch)
        optimizer.step(loss)

        if idx % 10 == 0:
            print('Net1 Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))


    return train_loss

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x


def val(model, val_loader1, val_loader2,epoch,rate,rate2):
    model.eval()
    ss = {}
    zz = {}
    for d1,d2 in zip(enumerate(val_loader1), enumerate(val_loader2)):
        ppp=0.1
        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]

        outputs = model(input1, input2, False, 1, 1, 0.5, 1)

        o1 = outputs[4][6]
        o2 = outputs[4][7]
        i1_s, i2_s, i1_z, i2_z = outputs[4][0], outputs[4][1], outputs[4][2], outputs[4][3]
        i1_s_mean, i1_s_log_var, i1_ss, i1s_lab = outputs[3][0], outputs[3][1], outputs[3][2], outputs[3][3]
        i1_z_mean, i1_z_log_var, i1_zz, i1z_lab = outputs[0][0], outputs[0][1], outputs[0][2], outputs[0][3]
        im1 = (o1[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i1/" + "i1_" + str(idx) + ".jpg", im1)

        im2 = (o2[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i2/" + "i2_" + str(idx) + ".jpg", im2)

        im_i1_s = (i1_s[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i1s/" + "i1s_" + str(idx) + ".jpg", im_i1_s)

        im_i2_s = (i2_s[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i2s/" + "i2s_" + str(idx) + ".jpg", im_i2_s)

        im_i1_z = (i1_z[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i1z/" + "i1z_" + str(idx) + ".jpg", im_i1_z)

        im_i2_z = (i2_z[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i2z/" + "i2z_" + str(idx) + ".jpg", im_i2_z)

def train2(model1, model2, train_loader1, train_loader2, lossfun, optimizer, epoch):
    model1.eval()
    model2.train()
    train_loss = []

    # checkpoint_path = os.path.join('checkpoints', name)
    # checkpoint_name = os.path.join(checkpoint_path, name + '-OUR_NOATT.pkl')
    # os.makedirs(checkpoint_path, exist_ok=True)
    #
    # if args.checkpoint is not None:
    #     print("loading........")
    #     net.load(args.checkpoint)

    for d1,d2 in zip(enumerate(train_loader1), enumerate(train_loader2)):

        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]

        net1_out = model1(input1, input2, False, 0.001, 0.2, 0.001, 0.2)

        z_fr_net1 = net1_out[4][2]
        outputs = model2(input1, input2,False, 0.01, 1)  # 0.1 0.5 1 1
        loss = lossfun(outputs, input1, input2,z_fr_net1.detach())
        optimizer.step(loss)

        if idx % 10 == 0:
            print('Net2 Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))
    return train_loss

def val2(model1, model2, val_loader1, val_loader2,epoch):
    model1.eval()
    model2.eval()
    for d1,d2 in zip(enumerate(val_loader1), enumerate(val_loader2)):
        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]
        net1_out = model1(input1, input2, False, 0.001, 0.2, 0.001, 0.2)

        z_fr_net1 = net1_out[4][2]
        outputs = model2(input1, input2, False, 0.01, 1)
        # i1_z = to_img(outputs[1][1])
        # o2 = to_img(outputs[1][0])
        # i1_s = to_img(outputs[1][2])
        # i1_z = outputs[1][1]
        # o2 = outputs[1][0]
        # i1_s = outputs[1][2]
        outputs11, outputs22, out_zz, out_ss1, out_ss2 =  outputs[0][0],outputs[0][1],outputs[0][2],outputs[0][3],outputs[0][4]

        # print("out_ss1:",out_ss1)
        i1ss = jt.clamp(input1-z_fr_net1,0,1)

        is1 = (i1ss[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net2/i1ss/" + "i1ss_" + str(idx) + ".jpg", is1)

        im1 = (out_zz[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net2/i1z/" + "i1z_" + str(idx) + ".jpg", im1)

        im2 = (outputs11[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net2/out1/" + "out1_" + str(idx) + ".jpg", im2)

        im3 = (outputs22[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net2/out2/" + "out2_" + str(idx) + ".jpg", im3)

        im_i1_s = (out_ss1[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net2/i1s/" + "i1s_" + str(idx) + ".jpg", im_i1_s)

        im_i2_s = (out_ss2[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net2/i2s/" + "i2s_" + str(idx) + ".jpg", im_i2_s)

        if epoch % 50 == 0:
            checkpoint_path = os.path.join('checkpoints', 'XRK')
            checkpoint_name = os.path.join(checkpoint_path, str(epoch) + 'SU_final_data_net2.pkl')
            os.makedirs(checkpoint_path, exist_ok=True)
            model2.save(checkpoint_name)

def train3(model2, model3, train_loader1, train_loader2, lossfun, optimizer, epoch):
    model2.eval()
    model3.train()
    train_loss = []

    # checkpoint_path = os.path.join('checkpoints', name)
    # checkpoint_name = os.path.join(checkpoint_path, name + '-OUR_NOATT.pkl')
    # os.makedirs(checkpoint_path, exist_ok=True)
    #
    # if args.checkpoint is not None:
    #     print("loading........")
    #     net.load(args.checkpoint)

    for d1, d2 in zip(enumerate(train_loader1), enumerate(train_loader2)):

        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]

        outputs2 = model2(input1, input2,False, 0.01, 1)  # 0.1 0.5 1 1

        outputs3 = model3(input1, input2,False, 0.01, 1)  # 0.1 0.5 1 1
        s1_fr_net1 = outputs2[0][3]
        s2_fr_net1 = outputs2[0][4]
        loss = lossfun(outputs3, input1, input2, s1_fr_net1,s2_fr_net1)
        optimizer.step(loss)

        if idx % 10 == 0:
            print('Net3 Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))
    return train_loss

def val3(model2, model3, val_loader1, val_loader2,epoch):
    model2.eval()
    model3.eval()
    for d1, d2 in zip(enumerate(val_loader1), enumerate(val_loader2)):
        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]
        outputs = model3(input1, input2, False, 0.01, 1)

        outputs11, outputs22, out_z1,out_z2, out_ss1, out_ss2 = outputs[4][4], outputs[4][5], outputs[4][2], outputs[4][3], \
                                                         outputs[4][0],outputs[4][1]




        iz1 = (out_z1[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net3/i1z/" + "i1z_" + str(idx) + ".jpg", iz1)

        iz2 = (out_z2[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net3/i2z/" + "i2z_" + str(idx) + ".jpg", iz2)

        im2 = (outputs11[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net3/out1/" + "out1_" + str(idx) + ".jpg", im2)

        im3 = (outputs22[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net3/out2/" + "out2_" + str(idx) + ".jpg", im3)

        im_i1_s = (out_ss1[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net3/i1s/" + "i1s_" + str(idx) + ".jpg", im_i1_s)

        im_i2_s = (out_ss2[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net3/i2s/" + "i2s_" + str(idx) + ".jpg", im_i2_s)

        if epoch % 50 == 0:
            checkpoint_path = os.path.join('checkpoints', 'XRK')
            checkpoint_name = os.path.join(checkpoint_path, str(epoch) + 'SU_final_data_net3.pkl')
            os.makedirs(checkpoint_path, exist_ok=True)
            model3.save(checkpoint_name)




def train4(model3, model4, train_loader1, train_loader2, lossfun, optimizer, epoch):
    model3.eval()
    model4.train()
    train_loss = []
    for d1,d2 in zip(enumerate(train_loader1), enumerate(train_loader2)):

        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]

        net1_out = model3(input1, input2, False, 0.001, 0.2)

        z1_fr_net1 = net1_out[4][2]
        z2_fr_net1 = net1_out[4][3]
        outputs = model4(input1, input2,False, 0.01, 1)  # 0.1 0.5 1 1
        loss = lossfun(outputs, input1, input2,z1_fr_net1.detach(),z2_fr_net1.detach())
        optimizer.step(loss)

        if idx % 10 == 0:
            print('Net4 Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))
    return train_loss

def val4(model3, model4, val_loader1, val_loader2,epoch):
    model3.eval()
    model4.eval()
    for d1,d2 in zip(enumerate(val_loader1), enumerate(val_loader2)):
        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]
        # net1_out = model4(input1, input2, False, 0.001, 0.2)

        # z_fr_net1 = net1_out[4][2]
        outputs = model4(input1, input2, False, 0.01, 1)

        outputs11, outputs22, out_z1, out_z2, out_ss1, out_ss2 = outputs[4][4], outputs[4][5], outputs[4][2], \
                                                                 outputs[4][3], \
                                                                 outputs[4][0], outputs[4][1]

        # print("out_ss1:",out_ss1)
        # i1ss = jt.clamp(input1-z_fr_net1,0,1)

        # is1 = (i1ss[0].numpy().transpose(1, 2, 0)) * 255
        # cv2.imwrite("./recon_net2/i1ss/" + "i1ss_" + str(idx) + ".jpg", is1)

        iz1 = (out_z1[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net4/i1z/" + "i1z_" + str(idx) + ".jpg", iz1)

        iz2 = (out_z2[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net4/i2z/" + "i2z_" + str(idx) + ".jpg", iz2)

        im2 = (outputs11[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net4/out1/" + "out1_" + str(idx) + ".jpg", im2)

        im3 = (outputs22[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net4/out2/" + "out2_" + str(idx) + ".jpg", im3)

        im_i1_s = (out_ss1[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net4/i1s/" + "i1s_" + str(idx) + ".jpg", im_i1_s)

        im_i2_s = (out_ss2[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon_net4/i2s/" + "i2s_" + str(idx) + ".jpg", im_i2_s)

        if epoch % 50 == 0:
            checkpoint_path = os.path.join('checkpoints', 'XRK')
            checkpoint_name = os.path.join(checkpoint_path, str(epoch) + 'SU_final_data_net4_l4.pkl')
            os.makedirs(checkpoint_path, exist_ok=True)
            model4.save(checkpoint_name)








rates = 0.2
rates2 = 1
# p= model.parameters()
# print(p['decoder_f.Li1.weight'])
import pandas as pd
for i in range(800):
    if i<200:
        a= train(model,train_loader1,train_loader2,loss,optimizer,i,rates,rates2)
        # with open("./loss/loss_"+str(i)+".txt","w") as f:
        #     for j in a:
        #         f.write(str(j)+"\n")
        val(model,val_loader1,val_loader2,i, rates,rates2)
        # if i==0 or i ==5 or i==15 or i==25 or i==35:
        #
        #     df1 = pd.DataFrame(p[72].data)
        #
        #     with pd.ExcelWriter('./mu_sigma/'+str(i)+'_v3.xlsx') as writer:
        #         df1.to_excel(writer, sheet_name='Sheet1')

    elif i<400 and i>=200:

        train2(model, model2, train_loader3, train_loader4, loss2, optimizer2, i)
        val2(model, model2, val_loader3, val_loader4,i)
        # train2(model, model2, train_loader3, train_loader3, loss2, optimizer2, i)
        # val2(model, model2, val_loader3, val_loader4)
        # with open("./loss/loss_" + str(i) + ".txt", "w") as f:
        #     for j in a:
        #         f.write(str(j) + "\n")

    elif i<500 and i>=400:
        train3(model2, model3, train_loader5, train_loader6, loss3, optimizer3, i)
        val3(model2, model3, val_loader5, val_loader6, i)
    else:
        train4(model3, model4, train_loader1, train_loader2, loss4, optimizer4, i)
        val4(model3, model4, val_loader1, val_loader2, i)
