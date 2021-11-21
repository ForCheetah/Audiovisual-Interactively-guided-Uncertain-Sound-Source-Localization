#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from options.train_options import TrainOptions
from data.data_loader import createDataloader
from models.models import ModelBuilder
from models.Localization_Current_Model import LocalizationModel
import torch
from torch.autograd import Variable
from utils import utils, viz
from models import criterion
import torch.nn.functional as F
from utils.SaveObject import SaveObject


def create_optimizer(nets, opt):
    (net_image_extract, net_audio_extract, net_AVattention, net_Audio_Separation,
     net_Align, net_ClassifyOnAudio, net_ClassifyOnImage) = nets
    param_groups = [
        {'params': net_image_extract.parameters(), 'lr': 0.00001},
        {'params': net_audio_extract.parameters(), 'lr': 0.00001},
        {'params': net_AVattention.parameters(), 'lr': 0.00001},
        {'params': net_Audio_Separation.parameters(), 'lr': 0.00001},
        {'params': net_Align.parameters(), 'lr': 0.00001},
        {'params': net_ClassifyOnAudio.parameters(), 'lr': 0.00001},
        {'params': net_ClassifyOnImage.parameters(), 'lr': 0.00001}
    ]
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor



# parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")
opt.stage = 'two'

# construct data loader
data_loader = createDataloader(mode='train', opt=opt)

print('#training images = %d' % len(data_loader))

# create validation set data loader if validation_on option is set
if opt.validation_on:
    # temperally set to val to load val data
    opt.mode = 'val'
    data_loader_val = createDataloader(mode='val', opt=opt)
    print('#validation images = %d' % len(data_loader_val))
    opt.mode = 'train'  # set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# Network Builders
builder = ModelBuilder()
net_image_extract = builder.build_ImageExtract(
    weights="")
net_audio_extract = builder.build_AudioExtract(
    weights="")
net_AVattention = builder.build_Heatmap(
    weights="")
net_Audio_Separation = builder.build_AudioSeparation(
    weights="")
net_Align = builder.build_Align(
    mode='train',
    weights="")
net_ClassifyOnAudio = builder.build_ClassifyOnAudio(
        weights="",
        num_class=opt.num_class)
net_ClassifyOnImage = builder.build_ClassifyOnImage(
        weights="",
        num_class=opt.num_class)


nets = (net_image_extract, net_audio_extract, net_AVattention,
        net_Audio_Separation, net_Align, net_ClassifyOnAudio, net_ClassifyOnImage)

# construct our audio-visual model
model = LocalizationModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)

# Set up optimizer
optimizer = create_optimizer(nets, opt)

# Set up loss functions
unsuperviseloss = criterion.InfoNCE()
loss_align = criterion.SimilarityLoss()
loss_imagePC = criterion.CEOfImage()
loss_audioPC = criterion.CEOfAudio()
utils.mkdirs(os.path.join('.', opt.checkpoints_dir, opt.name))

# initialization
total_batches = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_unsupervise_loss = []
batch_align_loss = []
batch_imagePC_loss = []
batch_audioPC_loss = []
best_err = float("inf")
totalNum = 0

for epoch in range(1 + opt.epoch_count, opt.epoch + 1):
    print('epoch:' + str(epoch))
    torch.cuda.synchronize()
    epoch_start_time = time.time()

    if (opt.measure_time):
        iter_start_time = time.time()
    for i, data in enumerate(data_loader):
        if total_batches % 10 == 0:
            print('epoch' + str(epoch) + '____' + str(total_batches))
        if (opt.measure_time):
            torch.cuda.synchronize()
            iter_data_loaded_time = time.time()

        # print(data['label'].size())
        total_batches += 1

        # forward pass
        model.zero_grad()
        output = model.forward(data)

# =================================================================================================
        # save the objects when needed
        # totalNum += len(data['image'])
        # print('current num total  :   ' + str(totalNum))
        # SaveObject(output, data['vid'])


        simLoss = unsuperviseloss(output['sim1'], output['sim2'], output['s_others'])
        alignLoss = loss_align(output['simi1'], output['simi2'], output['mask1'], output['mask2'])
        imagePCLoss_1 = loss_imagePC(output['imagePC1'], Variable(output['pseudo_label1'], requires_grad=False))
        imagePCLoss_2 = loss_imagePC(output['imagePC2'], Variable(output['pseudo_label2'], requires_grad=False))
        audioPCLoss_1 = loss_audioPC(output['audioPC1'], Variable(output['pseudo_label1'], requires_grad=False))
        audioPCLoss_2 = loss_audioPC(output['audioPC2'], Variable(output['pseudo_label2'], requires_grad=False))
        audioPCLoss = (audioPCLoss_1 + audioPCLoss_2)/2
        imagePCLoss = (imagePCLoss_1 + imagePCLoss_2)/2

        if (opt.measure_time):
            torch.cuda.synchronize()
            iter_data_forwarded_time = time.time()
        # store losses for this batch
        batch_unsupervise_loss.append(simLoss.item())
        batch_align_loss.append(alignLoss.item())
        batch_imagePC_loss.append(imagePCLoss.item())
        batch_audioPC_loss.append((audioPCLoss_1.item()+audioPCLoss_2.item())/2)

        optimizer.zero_grad()
        loss = simLoss + alignLoss * 0.6 + imagePCLoss * 0.3 + audioPCLoss * 0.3
        loss.backward()
        optimizer.step()

        if (opt.measure_time):
            torch.cuda.synchronize()
            iter_model_backwarded_time = time.time()

        if (opt.measure_time):
            torch.cuda.synchronize()
            iter_model_backwarded_time = time.time()
            data_loading_time.append(iter_data_loaded_time - iter_start_time)
            model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
            model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

        if (total_batches % opt.display_freq == 0):
            print('Display training progress at (epoch %d, total_batches %d)' % (epoch, total_batches))
            avg_unsupervise_loss = sum(batch_unsupervise_loss) / len(batch_unsupervise_loss)
            avg_align_loss = sum(batch_align_loss) / len(batch_align_loss)
            avg_imagePC_loss = sum(batch_imagePC_loss) / len(batch_imagePC_loss)
            avg_audioPC_loss = sum(batch_audioPC_loss) / len(batch_audioPC_loss)

            print('unsupervise loss: %.4f, align loss: %.4f, imagePC loss: %.4f, audioPC loss: %.4f' \
                  % (avg_unsupervise_loss, avg_align_loss, avg_imagePC_loss, avg_audioPC_loss))
            batch_unsupervise_loss = []
            batch_align_loss = []
            batch_ce_loss = []
            batch_audioPC_loss = []
            batch_imagePC_loss = []

            if opt.tensorboard:
                writer.add_scalar('data/simloss', avg_unsupervise_loss, total_batches)
                writer.add_scalar('data/align_loss', avg_align_loss, total_batches)
                writer.add_scalar('data/imagePC', avg_imagePC_loss, total_batches)
                writer.add_scalar('data/audioPC', avg_audioPC_loss, total_batches)

            if (opt.measure_time):
                print('average data loading time: %.3f' % (sum(data_loading_time) / len(data_loading_time)))
                print('average forward time: %.3f' % (sum(model_forward_time) / len(model_forward_time)))
                print('average backward time: %.3f' % (sum(model_backward_time) / len(model_backward_time)))
                data_loading_time = []
                model_forward_time = []
                model_backward_time = []
            print('end of display \n')

        if (total_batches % opt.save_latest_freq == 0):
            print('saving the latest model (epoch %d, total_batches %d)' % (epoch, total_batches))
            torch.save(net_audio_extract.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_extract_latest.pth'))
            torch.save(net_image_extract.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'image_extract_latest.pth'))
            torch.save(net_AVattention.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'AVattention_latest.pth'))
            torch.save(net_Audio_Separation.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audioSeparation_latest.pth'))
            torch.save(net_Align.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'align_latest.pth'))
            torch.save(net_ClassifyOnAudio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'ClassifyOnAudio_latest.pth'))
            torch.save(net_ClassifyOnImage.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'ClassifyOnImage_latest.pth'))

        # decrease learning rate
        if (total_batches in opt.lr_steps):
            decrease_learning_rate(optimizer, opt.decay_factor)
            print('decreased learning rate by ', opt.decay_factor)

        if (opt.measure_time):
            torch.cuda.synchronize()
            iter_start_time = time.time()
