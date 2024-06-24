import os

import cv2
from jittor.dataset.dataset import ImageFolder
import jittor.transform as transform
import jittor as jt
import jittor.nn as nn
import numpy as np
from CVAE_exp import CVAE

from PIL import Image
import random
from jittor.optim import Adam
import scipy.stats
from scipy.spatial.distance import pdist
np.random.seed(0)
random.seed(0)
# batch_size = 10
batch_size = 40
transform1 = transform.Compose([

    # transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transform.ToTensor()
    #   transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3])
    # transform.ToTensor()
])

train_i1_dir = './data/xrkexp/i1_train'
train_i2_dir = './data/xrkexp/i2_train'
train_loader1 = ImageFolder(train_i1_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
train_loader2 = ImageFolder(train_i2_dir).set_attrs(transform=transform1, batch_size=batch_size, shuffle=True)
val_i1_dir = './data/xrkexp/i1_eval'
val_i2_dir = './data/xrkexp/i2_eval'
val_loader1 = ImageFolder(val_i1_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)
val_loader2 = ImageFolder(val_i2_dir).set_attrs(transform=transform1, batch_size=1, shuffle=False)




def loss(out, inputs1, inputs2):
    i1_z_mean, i1_z_log_var, i1_z= out[0][0], out[0][1], out[0][2]
    i2_z_mean, i2_z_log_var, i2_z = out[1][0], out[1][1], out[1][2]

    i1_s_mean, i1_s_log_var, i1_s= out[2][0], out[2][1], out[2][2]


    outputs1, outputs2, q_score, q_bar_score = out[3][0], out[3][1], out[4][0], out[4][1]

    tc_loss = jt.log(q_score / (1 - q_score))
    discriminator_loss = - jt.log(q_score) - jt.log(1 - q_bar_score)



    reconstruction_loss = nn.mse_loss(jt.flatten(outputs1), jt.flatten(inputs1)) * 10000
    reconstruction_loss += nn.mse_loss(jt.flatten(outputs2), jt.flatten(inputs2)) * 10000

    kl_loss = 1 + i1_z_log_var - jt.pow(i1_z_mean,2) - jt.exp(i1_z_log_var)

    kl_loss += (1 + i1_s_log_var - jt.pow(i1_s_mean, 2) - jt.exp(i1_s_log_var))

    kl_loss += 1 + i2_z_log_var - jt.pow(i2_z_mean,2) - jt.exp(i2_z_log_var)




    kl_loss = kl_loss.sum(dim=-1)
    kl_loss *= -0.5
    # print("kl_loss", kl_loss)
    cvae_loss = jt.mean(reconstruction_loss + kl_loss + 80*tc_loss + discriminator_loss)


    return cvae_loss



jt.flags.use_cuda = 1
lr = 0.001
lr2 = 0.001
train_epoch = 10000
img_size = 100
momentum = 0.9
weight_decay = 1e-5
model = CVAE()


optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



def train(model, train_loader1, train_loader2, lossfun, optimizer, epoch):
    model.train()

    for d1,d2 in zip(enumerate(train_loader1), enumerate(train_loader2)):
        idx = d1[0]
        input1 = d1[1][0]


        input2 = d2[1][0]


        outputs = model(input1, input2, True)  # 0.1 0.5 1 1

        loss = lossfun(outputs, input1, input2)
        optimizer.step(loss)

        if idx % 10 == 0:
            print('Net Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))



def val(model, val_loader1, val_loader2,epoch):
    model.eval()
    ss = {}
    zz = {}
    for d1,d2 in zip(enumerate(val_loader1), enumerate(val_loader2)):

        idx = d1[0]
        input1 = d1[1][0]
        input2 = d2[1][0]

        outputs = model(input1, input2, False)

        o1 = outputs[3][0]
        o2 = outputs[3][1]
        i1_s = outputs[5]
        i1_z = outputs[6]
        # i1_s_mean, i1_s_log_var, i1_ss= outputs[3][0], outputs[3][1], outputs[3][2]
        # i1_z_mean, i1_z_log_var, i1_zz= outputs[0][0], outputs[0][1], outputs[0][2]

        im1 = (o1[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i1/" + "i1_" + str(idx) + ".jpg", im1)

        im2 = (o2[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i2/" + "i2_" + str(idx) + ".jpg", im2)

        im_i1_s = (i1_s[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i1s/" + "i1s_" + str(idx) + ".jpg", im_i1_s)

        im_i1_s = (i1_z[0].numpy().transpose(1, 2, 0)) * 255
        cv2.imwrite("./recon/i1z/" + "i1z_" + str(idx) + ".jpg", im_i1_s)



        # im_i1_z = (i1_z[0].numpy().transpose(1, 2, 0)) * 255
        # cv2.imwrite("./recon/i1z/" + "i1z_" + str(idx) + ".jpg", im_i1_z)
        #
        # im_i2_z = (i2_z[0].numpy().transpose(1, 2, 0)) * 255
        # cv2.imwrite("./recon/i2z/" + "i2z_" + str(idx) + ".jpg", im_i2_z)

        if epoch % 50 == 0:
            checkpoint_path = os.path.join('checkpoints', 'SU_exp')
            checkpoint_name = os.path.join(checkpoint_path, str(epoch) + 'SU_XRK_exp.pkl')
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save(checkpoint_name)

rates = 0.2
rates2 = 1

for i in range(15000):
    if i<4000:
        train(model,train_loader1,train_loader2,loss,optimizer,i)
        val(model,val_loader1,val_loader2,i)

