#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import imp
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from common.avgmeter import *
from common.logger import Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextInterpolate import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.SalsaNextCompletion import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoder import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderStageTwo import *
from tasks.semantic.modules.SalsaNextCompletionStageTwo import *
from tasks.semantic.modules.pncnn import *
from tasks.semantic.modules.SalsaNextCompletionStageTwoFilteredWeights import *
from tasks.semantic.modules.SalsaNextCompletionAttention import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderMaskPredictionHead import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderStageTwoPerceptionLoss import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderStageTwoEndToEnd import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderMaskPredictionHeadStageTwo import *

from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax, MaskedProbExpLoss
import tasks.semantic.modules.adf as adf

def keep_variance_fn(x):
    return x + 1e-3

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]

    return y_true


class SoftmaxHeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)

    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = torch.nn.functional.one_hot(targets, num_classes=20).permute(0,3,1,2).float()

        precision = 1 / (var + eps)
        return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SalsaNext" + suffix)


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None,uncertainty=False,completion=False,completion_single_decoder=False, completion_single_decoder_stage_two=False,completion_stage_two=False, pncnn=False,
                 pncnn_stage_two=False,
                 ipbasic=False,
                 bilateral_filtering=False,
                 interpolate=False,
                 completion_stage_two_filtered=False,
                 attn=False,
                 full_res=False,
                 edge_loss=False,
                 completion_single_decoder_mask_prediction=False,
                 completion_single_decoder_stage_two_perception_loss=False,
                 oracle=False,
                 completion_single_decoder_end_to_end=False,
                 completion_single_decoder_mask_prediction_stage_two=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.uncertainty = uncertainty
        self.completion_stage_two = completion_stage_two
        self.completion = completion
        self.completion_single_decoder_stage_two = completion_single_decoder_stage_two
        self.completion_single_decoder = completion_single_decoder
        self.pncnn = pncnn
        self.pncnn_stage_two = pncnn_stage_two
        self.ipbasic = ipbasic
        self.bilateral_filtering = bilateral_filtering
        self.interpolate = interpolate
        self.completion_stage_two_filtered = completion_stage_two_filtered
        self.attn = attn
        self.full_res = full_res
        self.edge_loss = edge_loss
        self.completion_single_decoder_mask_prediction = completion_single_decoder_mask_prediction
        self.completion_single_decoder_stage_two_perception_loss = completion_single_decoder_stage_two_perception_loss
        self.oracle = oracle
        self.completion_single_decoder_end_to_end = completion_single_decoder_end_to_end
        self.completion_single_decoder_mask_prediction_stage_two = completion_single_decoder_mask_prediction_stage_two

        if self.completion:
            self.multi_sensor = True
        if self.attn:
            self.multi_sensor = True
        elif self.completion_single_decoder:
            self.multi_sensor = True
        elif self.completion_single_decoder_mask_prediction:
            self.multi_sensor = True
        elif self.completion_single_decoder_end_to_end:
            self.multi_sensor = True
        elif self.completion_single_decoder_stage_two:
            self.multi_sensor = False
        elif self.completion_single_decoder_mask_prediction_stage_two:
            self.multi_sensor = False
        elif self.completion_stage_two:
            self.multi_sensor = False
        elif self.completion_stage_two_filtered:
            self.multi_sensor = False
        elif self.pncnn:
            self.multi_sensor = False
        elif self.pncnn_stage_two:
            self.multi_sensor = False
        elif self.ipbasic:
            self.multi_sensor = False
        elif self.bilateral_filtering:
            self.multi_sensor = False
        elif self.interpolate:
            self.multi_sensor = False # CHANGE TO FALSE
        elif self.completion_single_decoder_stage_two_perception_loss:
            self.multi_sensor = False
        else:
            self.multi_sensor = False

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_update": 0,
                     "train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        parserModule = imp.load_source("parserModule",
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/parser.py')
        if self.pncnn_stage_two or self.ipbasic:
            self.parser = parserModule.Parser(root=self.datadir,
                                              train_sequences=self.DATA["split"]["train"],
                                              valid_sequences=self.DATA["split"]["valid"],
                                              test_sequences=None,
                                              labels=self.DATA["labels"],
                                              color_map=self.DATA["color_map"],
                                              learning_map=self.DATA["learning_map"],
                                              learning_map_inv=self.DATA["learning_map_inv"],
                                              sensor=self.ARCH["dataset"]["sensor"],
                                              max_points=self.ARCH["dataset"]["max_points"],
                                              batch_size=self.ARCH["train"]["batch_size"],
                                              workers=self.ARCH["train"]["workers"],
                                              gt=True,
                                              shuffle_train=True,
                                              simple=False,
                                              multi_sensor=self.multi_sensor,
                                              no_transform=True)
        elif self.oracle:
            self.parser = parserModule.Parser(root=self.datadir,
                                              train_sequences=self.DATA["split"]["train"],
                                              valid_sequences=self.DATA["split"]["valid"],
                                              test_sequences=None,
                                              labels=self.DATA["labels"],
                                              color_map=self.DATA["color_map"],
                                              learning_map=self.DATA["learning_map"],
                                              learning_map_inv=self.DATA["learning_map_inv"],
                                              sensor=self.ARCH["dataset"]["sensor"],
                                              max_points=self.ARCH["dataset"]["max_points"],
                                              batch_size=self.ARCH["train"]["batch_size"],
                                              workers=self.ARCH["train"]["workers"],
                                              gt=True,
                                              shuffle_train=True,
                                              simple=False,
                                              multi_sensor=self.multi_sensor,
                                              oracle=True)
        else:
            self.parser = parserModule.Parser(root=self.datadir,
                                              train_sequences=self.DATA["split"]["train"],
                                              valid_sequences=self.DATA["split"]["valid"],
                                              test_sequences=None,
                                              labels=self.DATA["labels"],
                                              color_map=self.DATA["color_map"],
                                              learning_map=self.DATA["learning_map"],
                                              learning_map_inv=self.DATA["learning_map_inv"],
                                              sensor=self.ARCH["dataset"]["sensor"],
                                              max_points=self.ARCH["dataset"]["max_points"],
                                              batch_size=self.ARCH["train"]["batch_size"],
                                              workers=self.ARCH["train"]["workers"],
                                              gt=True,
                                              shuffle_train=True,
                                              simple=False,
                                              multi_sensor=self.multi_sensor)

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            if self.uncertainty:
                self.model = SalsaNextUncertainty(self.parser.get_n_classes())
            elif self.completion_single_decoder:
                self.model = SalsaNextCompletionSingleDecoder(self.parser.get_n_classes())
            elif self.completion_single_decoder_mask_prediction:
                self.model = SalsaNextCompletionSingleDecoderMaskPredictionHead(self.parser.get_n_classes())
            elif self.completion:
                self.model = SalsaNextCompletion(self.parser.get_n_classes())
            elif self.attn:
                self.model = SalsaNextCompletionAttention(self.parser.get_n_classes())
            elif self.completion_single_decoder_stage_two:
                self.model = SalsaNextCompletionSingleDecoderStageTwo(self.parser.get_n_classes())
                if self.path is None:
                    pretrained_dict_path = '../../../log_test_completion/logs/single_decoder_stage_1_hd'
                    torch.nn.Module.dump_patches = True
                    w_dict = torch.load(pretrained_dict_path + "/SalsaNext",
                                        map_location=lambda storage, loc: storage)
                    self.model.completion_network.load_state_dict(w_dict['state_dict'], strict=True)
                    # model_dict = self.model.state_dict()
                    # pretrained_dict = torch.load(pretrained_dict_path + "/SalsaNext",
                    #                     map_location=lambda storage, loc: storage)['state_dict']
                    #
                    # # 1. filter out unnecessary keys
                    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    # # 2. overwrite entries in the existing state dict
                    # model_dict.update(pretrained_dict)
                    # # 3. load the new state dict
                    # self.model.completion_network.load_state_dict(model_dict)
            elif self.completion_single_decoder_mask_prediction_stage_two:
                self.model = SalsaNextCompletionSingleDecoderMaskPredictionHeadStageTwo(self.parser.get_n_classes())
                if self.path is None:
                    pretrained_dict_path = '../../../log_test_completion/logs/single_decoder_mask_prediction_edge_loss_stage_1_hd'
                    torch.nn.Module.dump_patches = True
                    w_dict = torch.load(pretrained_dict_path + "/SalsaNext",
                                        map_location=lambda storage, loc: storage)
                    self.model.completion_network.load_state_dict(w_dict['state_dict'], strict=True)
            elif self.completion_single_decoder_stage_two_perception_loss:
                self.model = SalsaNextCompletionSingleDecoderStageTwoPerceptionLoss(self.parser.get_n_classes())
                if self.path is None:
                    pretrained_dict_path = '../../../log_test_completion/logs/single_decoder_stage_1_hd'
                    torch.nn.Module.dump_patches = True
                    w_dict = torch.load(pretrained_dict_path + "/SalsaNext",
                                        map_location=lambda storage, loc: storage)
                    self.model.completion_network.load_state_dict(w_dict['state_dict'], strict=True)
            elif self.completion_stage_two:
                self.model = SalsaNextCompletionStageTwo(self.parser.get_n_classes())
                if self.path is None:
                    pretrained_dict_path = '../../../log_test_completion/logs/dual_decoder_stage_1_hd'
                    torch.nn.Module.dump_patches = True
                    w_dict = torch.load(pretrained_dict_path + "/SalsaNext",
                                        map_location=lambda storage, loc: storage)
                    self.model.completion_network.load_state_dict(w_dict['state_dict'], strict=True)
            elif self.completion_stage_two_filtered:
                self.model = SalsaNextCompletionStageTwoFilteredWeights(self.parser.get_n_classes())
                if self.path is None:
                    pretrained_dict_path = '../../../log_test_completion/logs/dual_decoder_stage_1_hd'
                    torch.nn.Module.dump_patches = True
                    w_dict = torch.load(pretrained_dict_path + "/SalsaNext",
                                        map_location=lambda storage, loc: storage)
                    self.model.completion_network.load_state_dict(w_dict['state_dict'], strict=True)
            elif self.completion_single_decoder_end_to_end:
                self.model = SalsaNextCompletionSingleDecoderStageTwoEndToEnd(self.parser.get_n_classes())
            elif self.pncnn:
                self.model = PNCNN(self.parser.get_n_classes())
            elif self.pncnn_stage_two:
                self.model = SalsaNext(self.parser.get_n_classes())
            elif self.interpolate:
                self.model = SalsaNextInterpolate(self.parser.get_n_classes())
            else:
                self.model = SalsaNext(self.parser.get_n_classes())
        torch.autograd.set_detect_anomaly(True)
        self.tb_logger = Logger(self.log + "/tb")

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

        if self.pncnn:
            self.criterion = MaskedProbExpLoss().to(self.device)
        else:
            self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        if self.edge_loss and not self.completion_single_decoder_mask_prediction:
            self.l1 = nn.L1Loss(reduction='none').to(self.device)
        else:
            self.l1 = nn.L1Loss().to(self.device)
        if self.edge_loss:
            self.bce = nn.BCELoss(reduction='none').to(self.device)
        else:
            self.bce = nn.BCELoss().to(self.device)
        self.SoftmaxHeteroscedasticLoss = SoftmaxHeteroscedasticLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.l1 = nn.DataParallel(self.l1).cuda()
            self.ls = nn.DataParallel(self.ls).cuda()
            self.SoftmaxHeteroscedasticLoss = nn.DataParallel(self.SoftmaxHeteroscedasticLoss).cuda()
        self.optimizer = optim.SGD([{'params': self.model.parameters()}],
                                   lr=self.ARCH["train"]["lr"],
                                   momentum=self.ARCH["train"]["momentum"],
                                   weight_decay=self.ARCH["train"]["w_decay"])

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=self.ARCH["train"]["lr"],
                                  warmup_steps=up_steps,
                                  momentum=self.ARCH["train"]["momentum"],
                                  decay=final_decay)

        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(path + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.optimizer.load_state_dict(w_dict['optimizer'])
            self.epoch = w_dict['epoch'] + 1
            self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
            self.info = w_dict['info']
            print("info", w_dict['info'])


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(input_depth, input_mask, pred_depth, depth, mask, pred, gt, color_fn, unfold_mask):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        input_depth = (cv2.normalize(input_depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            input_depth, Trainer.get_mpl_colormap('viridis')) * input_mask[..., None]
        # make range image (normalized to 0,1 for saving)
        pred_depth = (cv2.normalize(pred_depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        # unfold mask
        if unfold_mask is not None:
            out_img_1 = cv2.applyColorMap(
                pred_depth, Trainer.get_mpl_colormap('viridis')) * unfold_mask[0, ..., None]
        else:
            out_img_1 = cv2.applyColorMap(
                pred_depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        out_img = np.concatenate([out_img, out_img_1], axis=0)
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img_2 = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        out_img = np.concatenate([out_img, out_img_2], axis=0)

        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # # save summaries of weights and biases
        # if w_summary and model:
        #     for tag, value in model.named_parameters():
        #         tag = tag.replace('.', '/')
        #         logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        #         if value.grad is not None:
        #             logger.histo_summary(
        #                 tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def train(self):

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):

            # train for 1 epoch
            acc, iou, loss, l1_loss, bce_loss, update_mean,hetero_l = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                           model=self.model,
                                                           criterion=self.criterion,
                                                           optimizer=self.optimizer,
                                                           epoch=epoch,
                                                           evaluator=self.evaluator,
                                                           scheduler=self.scheduler,
                                                           color_fn=self.parser.to_color,
                                                           report=self.ARCH["train"]["report_batch"],
                                                           show_scans=self.ARCH["train"]["show_scans"])

            # update info
            self.info["train_update"] = update_mean
            self.info["train_loss"] = loss
            self.info["train_l1_loss"] = l1_loss
            self.info["train_bce_loss"] = bce_loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou
            self.info["train_hetero"] = hetero_l

            # remember best iou and save checkpoint
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="")

            if self.info['train_iou'] > self.info['best_train_iou']:
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img,hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         evaluator=self.evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou
                self.info['valid_heteros'] = hetero_l

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']

                # save the weights!
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

            # save to log
            Trainer.save_to_log(logdir=self.log,
                                logger=self.tb_logger,
                                info=self.info,
                                epoch=epoch,
                                w_summary=self.ARCH["train"]["save_summary"],
                                model=self.model_single,
                                img_summary=self.ARCH["train"]["save_scans"],
                                imgs=rand_img)

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        l1_l = AverageMeter()
        bce_l = AverageMeter()
        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()
        cul_in_vol = None
        cul_proj_labels = None

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()
        # switch to train mode
        model.train()

        end = time.time()
        for i, (in_vol, proj_labels_input, proj_mask, proj_labels_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
            # if i > 2:
            #     break
            # measure data loading time
            self.data_time_t.update(time.time() - end)
            if self.gpu:
                in_vol = in_vol.cuda()
                proj_mask = proj_mask.cuda()
                if self.pncnn or self.completion_single_decoder_mask_prediction:
                    proj_labels_mask = proj_labels_mask.cuda()
                proj_labels_input = proj_labels_input.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()

            # compute output
            if self.uncertainty:
                output = model(in_vol)
                output_mean, output_var = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)(*output)
                hetero = self.SoftmaxHeteroscedasticLoss(output,proj_labels)
                loss_m = criterion(output_mean.clamp(min=1e-8), proj_labels) + hetero + self.ls(output_mean, proj_labels.long())
                hetero_l.update(hetero.mean().item(), in_vol.size(0))
                output = output_mean
            elif self.completion:
                output, completed_scene = model(in_vol, proj_mask)
                mask = (proj_mask - 1) * (-1)
                # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                if self.edge_loss:
                    l1 = self.l1(completed_scene,
                                 proj_labels_input)
                    edge_mask = None
                    for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                        # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                        edges = cv2.Canny((proj_labels_mask[batch_idx_edge_map]*255).detach().cpu().numpy().astype(np.uint8), 100, 200)
                        edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                        # cv2.imwrite('edges_visualization.png', edges)
                        edges_torch = torch.from_numpy(edges/255*5).cuda()
                        if edge_mask is not None:
                            edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)),dim=0)
                        else:
                            edge_mask = edges_torch.unsqueeze(0)
                    l1 += l1 * edge_mask.unsqueeze(1)
                    l1 = l1.mean()
                else:
                    l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                 proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
                if loss_m.sum() == 0:
                    loss_m = torch.zeros((1)).cuda()
                loss_m += l1
            elif self.attn:
                output, completed_scene = model(in_vol, proj_mask)
                mask = (proj_mask - 1) * (-1)
                # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                if self.edge_loss:
                    l1 = self.l1(completed_scene,
                                 proj_labels_input)
                    edge_mask = None
                    for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                        # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                        edges = cv2.Canny((proj_labels_mask[batch_idx_edge_map]*255).detach().cpu().numpy().astype(np.uint8), 100, 200)
                        edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                        # cv2.imwrite('edges_visualization.png', edges)
                        edges_torch = torch.from_numpy(edges/255*5).cuda()
                        if edge_mask is not None:
                            edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)),dim=0)
                        else:
                            edge_mask = edges_torch.unsqueeze(0)
                    l1 += l1 * edge_mask.unsqueeze(1)
                    l1 = l1.mean()
                else:
                    l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                 proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
                if loss_m.sum() == 0:
                    loss_m = torch.zeros((1)).cuda()
                loss_m += l1
            elif self.completion_single_decoder:
                completed_scene = model(in_vol, proj_mask)
                mask = (proj_mask - 1) * (-1) # not on projection mask
                # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                if self.edge_loss:
                    l1 = self.l1(completed_scene,
                                 proj_labels_input)
                    edge_mask = None
                    for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                        # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                        edges = cv2.Canny((proj_labels_mask[batch_idx_edge_map]*255).detach().cpu().numpy().astype(np.uint8), 100, 200)
                        edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                        # cv2.imwrite('edges_visualization.png', edges)
                        edges_torch = torch.from_numpy(edges/255*5).cuda()
                        if edge_mask is not None:
                            edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)),dim=0)
                        else:
                            edge_mask = edges_torch.unsqueeze(0)
                    l1 += l1 * edge_mask.unsqueeze(1)
                    l1 = l1.mean()
                    # cv2.imwrite('edges_visualization.png', edges)
                else:
                    l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                 proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                loss_m = l1
            elif self.completion_single_decoder_mask_prediction:
                completed_scene, completed_mask = model(in_vol, proj_mask)
                mask = (proj_mask - 1) * (-1)  # not on projection mask
                l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                if self.edge_loss:
                    l1 = self.l1(completed_scene,
                                 proj_labels_input)
                    bce = self.bce(completed_mask, proj_labels_mask.float())
                    edge_mask = None
                    for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                        # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                        edges = cv2.Canny(
                            (proj_labels_mask[batch_idx_edge_map] * 255).detach().cpu().numpy().astype(np.uint8),
                            100, 200)
                        edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                        # cv2.imwrite('edges_visualization.png', edges)
                        edges_torch = torch.from_numpy(edges / 255 * 5).cuda()
                        if edge_mask is not None:
                            edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)), dim=0)
                        else:
                            edge_mask = edges_torch.unsqueeze(0)
                    bce += bce * edge_mask
                    bce = bce.mean()
                    # cv2.imwrite('edges_visualization.png', edges)
                else:
                    l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                 proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                    bce = self.bce(completed_mask, proj_labels_mask.float())
                loss_m = l1 + bce
            elif self.completion_single_decoder_stage_two:
                output = model(in_vol, proj_mask)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            elif self.completion_single_decoder_mask_prediction_stage_two:
                output = model(in_vol, proj_mask)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            elif self.completion_single_decoder_end_to_end:
                output, completed_scene = model(in_vol, proj_mask)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
                if not loss_m.numel():
                    print('haha')
                    loss_m = 0.
                mask = (proj_mask - 1) * (-1) # not on projection mask
                # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                if self.edge_loss:
                    l1 = self.l1(completed_scene,
                                 proj_labels_input)
                    edge_mask = None
                    for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                        # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                        edges = cv2.Canny((proj_labels_mask[batch_idx_edge_map]*255).detach().cpu().numpy().astype(np.uint8), 100, 200)
                        edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                        # cv2.imwrite('edges_visualization.png', edges)
                        edges_torch = torch.from_numpy(edges/255*5).cuda()
                        if edge_mask is not None:
                            edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)),dim=0)
                        else:
                            edge_mask = edges_torch.unsqueeze(0)
                    l1 += l1 * edge_mask.unsqueeze(1)
                    l1 = l1.mean()
                    # cv2.imwrite('edges_visualization.png', edges)
                else:
                    l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                 proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                loss_m += l1
            elif self.completion_single_decoder_stage_two_perception_loss:
                output, feats = model(in_vol, proj_mask)
                gt_feats_total = None
                for batch_idx_feats in range(feats.shape[0]):
                    gt_feats = np.load(os.path.join('/media/user/storage', 'fr_feats', path_seq[batch_idx_feats], path_name[batch_idx_feats].split('.')[0] + ".npy"))
                    gt_feats = torch.from_numpy(gt_feats).cuda()
                    if gt_feats_total is not None:
                        gt_feats_total = torch.cat((gt_feats_total, gt_feats.unsqueeze(0)),dim=0)
                    else:
                        gt_feats_total = gt_feats.unsqueeze(0)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
                l1 = self.l1(feats, gt_feats_total)
                loss_m += l1

            elif self.completion_stage_two:
                output = model(in_vol, proj_mask)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            elif self.completion_stage_two_filtered:
                output, completed_scene, uf_mask = model(in_vol, proj_mask)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())

            elif self.pncnn:
                in_vol[:, 0] += 1.6433
                in_vol[:, 1] += 6.2918
                in_vol[:, 2] += 7.0150
                in_vol[:, 3] += 4.9339
                in_vol[:, 4] += 0.6881
                proj_labels_input[:, 0] += 1.6433
                proj_labels_input[:, 1] += 6.2918
                proj_labels_input[:, 2] += 7.0150
                proj_labels_input[:, 3] += 4.9339
                proj_labels_input[:, 4] += 0.6881
                in_vol[:, 0] /= 1.6433 + 2.1008
                in_vol[:, 1] /= 6.2918 + 6.1954
                in_vol[:, 2] /= 7.0150 + 6.9538
                in_vol[:, 3] /= 4.9339 + 12.7296
                in_vol[:, 4] /= 0.6881 + 0.3313
                proj_labels_input[:, 0] /= 1.6433 + 2.1008
                proj_labels_input[:, 1] /= 6.2918 + 6.1954
                proj_labels_input[:, 2] /= 7.0150 + 6.9538
                proj_labels_input[:, 3] /= 4.9339 + 12.7296
                proj_labels_input[:, 4] /= 0.6881 + 0.3313
                in_vol *= proj_mask.unsqueeze(1)
                proj_labels_input *= proj_labels_mask.unsqueeze(1)
                output = model(in_vol)
                loss_m = criterion(output, proj_labels_input, proj_labels_mask)
                in_vol[:, 0] *= 1.6433 + 2.1008
                in_vol[:, 1] *= 6.2918 + 6.1954
                in_vol[:, 2] *= 7.0150 + 6.9538
                in_vol[:, 3] *= 4.9339 + 12.7296
                in_vol[:, 4] *= 0.6881 + 0.3313
                proj_labels_input[:, 0] *= 1.6433 + 2.1008
                proj_labels_input[:, 1] *= 6.2918 + 6.1954
                proj_labels_input[:, 2] *= 7.0150 + 6.9538
                proj_labels_input[:, 3] *= 4.9339 + 12.7296
                proj_labels_input[:, 4] *= 0.6881 + 0.3313
                output[:, 0] *= 1.6433 + 2.1008
                output[:, 1] *= 6.2918 + 6.1954
                output[:, 2] *= 7.0150 + 6.9538
                output[:, 3] *= 4.9339 + 12.7296
                output[:, 4] *= 0.6881 + 0.3313
                in_vol[:, 0] -= 1.6433
                in_vol[:, 1] -= 6.2918
                in_vol[:, 2] -= 7.0150
                in_vol[:, 3] -= 4.9339
                in_vol[:, 4] -= 0.6881
                output[:, 0] -= 1.6433
                output[:, 1] -= 6.2918
                output[:, 2] -= 7.0150
                output[:, 3] -= 4.9339
                output[:, 4] -= 0.6881
                proj_labels_input[:, 0] -= 1.6433
                proj_labels_input[:, 1] -= 6.2918
                proj_labels_input[:, 2] -= 7.0150
                proj_labels_input[:, 3] -= 4.9339
                proj_labels_input[:, 4] -= 0.6881
                in_vol *= proj_mask.unsqueeze(1)
                proj_labels_input *= proj_labels_mask.unsqueeze(1)
            elif self.pncnn_stage_two:
                completed_scene = None
                for batch_idx in range(len(path_seq)):
                    scan_file = os.path.join('/home/user/Documents/carla_data/pncnn/pncnn_output', path_seq[batch_idx], 'kitti_velodyne', path_name[batch_idx].replace('bin','npy'))
                    one_completed_scene = np.load(scan_file)
                    one_completed_scene = torch.from_numpy(one_completed_scene).clone().cuda()
                    if completed_scene is not None:
                        completed_scene = torch.cat((completed_scene, one_completed_scene),dim=0)
                    else:
                        completed_scene = one_completed_scene
                mask = (proj_mask - 1) * (-1)
                completed_scene = completed_scene * mask.unsqueeze(1) + in_vol

                output = model(in_vol)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            elif self.ipbasic:
                completed_scene = None
                for batch_idx in range(len(path_seq)):
                    scan_file = os.path.join('/home/user/Documents/carla_data/ip_basic/demos/outputs', path_seq[batch_idx], 'kitti_velodyne', path_name[batch_idx].replace('bin','npy'))
                    one_completed_scene = np.load(scan_file)
                    one_completed_scene = torch.from_numpy(one_completed_scene).clone().cuda()
                    if completed_scene is not None:
                        completed_scene = torch.cat((completed_scene, one_completed_scene.unsqueeze(0)),dim=0)
                    else:
                        completed_scene = one_completed_scene.unsqueeze(0)
                mask = (proj_mask - 1) * (-1)
                completed_scene = completed_scene * mask.unsqueeze(1) + in_vol

                output = model(in_vol)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            elif self.bilateral_filtering:
                completed_scene = None
                for batch_idx in range(len(path_seq)):
                    one_completed_scene = in_vol[batch_idx].detach().cpu().numpy()
                    one_completed_scene = cv2.normalize(one_completed_scene, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    # one_completed_scene_min = one_completed_scene.min()
                    # one_completed_scene_max = one_completed_scene.max()
                    # one_completed_scene -= one_completed_scene_min
                    # one_completed_scene /= one_completed_scene_max + one_completed_scene_max
                    one_completed_scene_completed = None
                    for image_idx in range(5):
                        one_completed_scene_image = cv2.bilateralFilter(one_completed_scene[image_idx], 5, 25, 25)
                        one_completed_scene_image = cv2.normalize(one_completed_scene_image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        one_completed_scene_image = torch.from_numpy(one_completed_scene_image).clone().cuda()
                        one_completed_scene_image = one_completed_scene_image.unsqueeze(0)
                        if one_completed_scene_completed is not None:
                            one_completed_scene_completed = torch.cat((one_completed_scene_completed, one_completed_scene_image), dim=0)
                        else:
                            one_completed_scene_completed = one_completed_scene_image
                    if completed_scene is not None:
                        completed_scene = torch.cat((completed_scene, one_completed_scene_completed.unsqueeze(0)),dim=0)
                    else:
                        completed_scene = one_completed_scene_completed.unsqueeze(0)
                output = model(completed_scene)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            elif self.interpolate:
                output, completed_scene = model(in_vol, proj_mask)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            elif self.full_res:
                output = model(proj_labels_input)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            else:
                output = model(in_vol)
                loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())

            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss_m.backward(idx)
            else:
                loss_m.backward(retain_graph=True)
            optimizer.step()

            # measure accuracy and record loss
            loss = loss_m.mean()
            losses.update(loss.item(), in_vol.size(0))
            if self.completion or self.completion_single_decoder or self.completion_single_decoder_mask_prediction or self.completion_single_decoder_stage_two_perception_loss or self.completion_single_decoder_end_to_end:
                l1_l.update(l1.mean().item(), in_vol.size(0))

            if self.completion_single_decoder_mask_prediction:
                bce_l.update(bce.mean().item(), in_vol.size(0))

            if not (self.completion_single_decoder or self.completion_single_decoder_mask_prediction or self.pncnn):
                with torch.no_grad():
                    evaluator.reset()
                    argmax = output.argmax(dim=1)
                    evaluator.addBatch(argmax, proj_labels)
                    accuracy = evaluator.getacc()
                    jaccard, class_jaccard = evaluator.getIoU()

                acc.update(accuracy.item(), in_vol.size(0))
                iou.update(jaccard.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                # get the first scan in batch and project points
                input_depth_np = in_vol[0][0].cpu().numpy()
                input_mask_np = proj_mask[0].cpu().numpy()
                mask_np = proj_labels_mask[0].cpu().numpy()
                depth_np = proj_labels_input[0][0].cpu().numpy()
                completed_scene_np = completed_scene[0][0].cpu().numpy()
                pred_np = argmax[0].cpu().numpy()
                gt_np = proj_labels[0].cpu().numpy()
                out = Trainer.make_log_img(input_depth_np, input_mask_np, completed_scene_np, depth_np, mask_np, pred_np, gt_np, color_fn)

                input_depth_np = in_vol[1][0].cpu().numpy()
                input_mask_np = proj_mask[1].cpu().numpy()
                mask_np = proj_labels_mask[1].cpu().numpy()
                completed_scene_np = completed_scene[1][0].cpu().numpy()
                depth_np = proj_labels_input[1][0].cpu().numpy()
                pred_np = argmax[1].cpu().numpy()
                gt_np = proj_labels[1].cpu().numpy()
                out2 = Trainer.make_log_img(input_depth_np, input_mask_np, completed_scene_np, depth_np, mask_np, pred_np, gt_np, color_fn)

                out = np.concatenate([out, out2], axis=0)
                cv2.imshow("sample_training", out)
                cv2.waitKey(1)
            if self.uncertainty:

                if i % self.ARCH["train"]["report_batch"] == 0:
                    print( 'Lr: {lr:.3e} | '
                          'Update: {umean:.3e} mean,{ustd:.3e} std | '
                          'Epoch: [{0}][{1}/{2}] | '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                          'Hetero {hetero_l.val:.4f} ({hetero_l.avg:.4f}) | '
                          'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                          'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, hetero_l=hetero_l,acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))

                    save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
                          'Update: {umean:.3e} mean,{ustd:.3e} std | '
                          'Epoch: [{0}][{1}/{2}] | '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                          'L1 Loss {l_one.val:.4f} ({l_one.avg:.4f}) | '
                          'BCE Loss {l_bce.val:.4f} ({l_bce.avg:.4f}) | '
                          'Hetero {hetero.val:.4f} ({hetero.avg:.4f}) | '
                          'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                          'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, l_one=l1_l, l_bce=bce_l, hetero=hetero_l,acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))
            else:
                if i % self.ARCH["train"]["report_batch"] == 0:
                    print('Lr: {lr:.3e} | '
                          'Update: {umean:.3e} mean,{ustd:.3e} std | '
                          'Epoch: [{0}][{1}/{2}] | '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                          'L1 Loss {l_one.val:.4f} ({l_one.avg:.4f}) | '
                          'BCE Loss {l_bce.val:.4f} ({l_bce.avg:.4f}) | '
                          'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                          'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, l_one=l1_l, l_bce=bce_l, acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))

                    save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
                                                     'Update: {umean:.3e} mean,{ustd:.3e} std | '
                                                     'Epoch: [{0}][{1}/{2}] | '
                                                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                                                     'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                                                     'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                                                     'L1 Loss {l_one.val:.4f} ({l_one.avg:.4f}) | '
                                                     'BCE Loss {l_bce.val:.4f} ({l_bce.avg:.4f}) | '
                                                     'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                                                     'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, l_one=l1_l, l_bce=bce_l, acc=acc, iou=iou, lr=lr,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i)))

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, l1_l.avg, bce_l.avg, update_ratio_meter.avg,hetero_l.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        l1s = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_labels_input, proj_mask, proj_labels_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(val_loader):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                    proj_labels_input = proj_labels_input.cuda()
                    if self.pncnn or self.completion_single_decoder_mask_prediction:
                        proj_labels_mask = proj_labels_mask.cuda()

                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # compute output
                if self.uncertainty:
                    log_var, output, _ = model(in_vol)
                    log_out = torch.log(output.clamp(min=1e-8))
                    mean = output.argmax(dim=1)
                    log_var = log_var.mean(dim=1)
                    hetero = self.SoftmaxHeteroscedasticLoss(mean.float(),proj_labels.float()).mean()
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                    hetero_l.update(hetero.mean().item(), in_vol.size(0))
                elif self.completion:
                    output, completed_scene = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    mask = (proj_mask - 1) * (-1)
                    # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                    if self.edge_loss:
                        l1 = self.l1(completed_scene,
                                     proj_labels_input)
                        edge_mask = None
                        for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                            # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                            edges = cv2.Canny(
                                (proj_labels_mask[batch_idx_edge_map] * 255).detach().cpu().numpy().astype(np.uint8),
                                100, 200)
                            edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                            # cv2.imwrite('edges_visualization.png', edges)
                            edges_torch = torch.from_numpy(edges / 255 * 5).cuda()
                            if edge_mask is not None:
                                edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)), dim=0)
                            else:
                                edge_mask = edges_torch.unsqueeze(0)
                        l1 += l1 * edge_mask.unsqueeze(1)
                        l1 = l1.mean()
                    else:
                        l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                     proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.attn:
                    output, completed_scene = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    mask = (proj_mask - 1) * (-1)
                    # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                    if self.edge_loss:
                        l1 = self.l1(completed_scene,
                                     proj_labels_input)
                        edge_mask = None
                        for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                            # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                            edges = cv2.Canny(
                                (proj_labels_mask[batch_idx_edge_map] * 255).detach().cpu().numpy().astype(np.uint8),
                                100, 200)
                            edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                            # cv2.imwrite('edges_visualization.png', edges)
                            edges_torch = torch.from_numpy(edges / 255 * 5).cuda()
                            if edge_mask is not None:
                                edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)), dim=0)
                            else:
                                edge_mask = edges_torch.unsqueeze(0)
                        l1 += l1 * edge_mask.unsqueeze(1)
                        l1 = l1.mean()
                    else:
                        l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                     proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc + l1
                elif self.completion_single_decoder_mask_prediction:
                    completed_scene, completed_mask = model(in_vol, proj_mask)
                    mask = (proj_mask - 1) * (-1)  # not on projection mask
                    l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                    if self.edge_loss:
                        # l1 = self.l1(completed_scene,
                        #              proj_labels_input)
                        bce = self.bce(completed_mask, proj_labels_mask.float())
                        edge_mask = None
                        for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                            # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                            edges = cv2.Canny(
                                (proj_labels_mask[batch_idx_edge_map] * 255).detach().cpu().numpy().astype(np.uint8),
                                100, 200)
                            edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                            # cv2.imwrite('edges_visualization.png', edges)
                            edges_torch = torch.from_numpy(edges / 255 * 5).cuda()
                            if edge_mask is not None:
                                edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)), dim=0)
                            else:
                                edge_mask = edges_torch.unsqueeze(0)
                        # l1 += l1 * edge_mask.unsqueeze(1)
                        # l1 = l1.mean()
                        bce += bce * edge_mask
                        bce = bce.mean()
                        # cv2.imwrite('edges_visualization.png', edges)
                    else:
                        l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                     proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                        bce = self.bce(completed_mask, proj_labels_mask.float())
                    loss = l1 + bce
                    jacc = 0.
                    wce = 0.
                elif self.completion_single_decoder:
                    completed_scene = model(in_vol, proj_mask)
                    mask = (proj_mask - 1) * (-1)
                    # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                    if self.edge_loss:
                        l1 = self.l1(completed_scene,
                                     proj_labels_input)
                        edge_mask = None
                        for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                            # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                            edges = cv2.Canny(
                                (proj_labels_mask[batch_idx_edge_map] * 255).detach().cpu().numpy().astype(np.uint8),
                                100, 200)
                            edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                            # cv2.imwrite('edges_visualization.png', edges)
                            edges_torch = torch.from_numpy(edges / 255 * 5).cuda()
                            if edge_mask is not None:
                                edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)), dim=0)
                            else:
                                edge_mask = edges_torch.unsqueeze(0)
                        l1 += l1 * edge_mask.unsqueeze(1)
                        l1 = l1.mean()
                    else:
                        l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                     proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                    loss = l1
                    jacc = 0.
                    wce = 0.
                elif self.completion_single_decoder_stage_two_perception_loss:
                    output, feats = model(in_vol, proj_mask)
                    gt_feats_total = None
                    for batch_idx_feats in range(feats.shape[0]):
                        gt_feats = np.load(os.path.join('/media/user/storage', 'fr_feats', path_seq[batch_idx_feats],
                                                        path_name[batch_idx_feats].split('.')[0] + ".npy"))
                        gt_feats = torch.from_numpy(gt_feats).cuda()
                        if gt_feats_total is not None:
                            gt_feats_total = torch.cat((gt_feats_total, gt_feats.unsqueeze(0)), dim=0)
                        else:
                            gt_feats_total = gt_feats.unsqueeze(0)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    l1 = self.l1(feats, gt_feats_total)
                    loss = wce + jacc + l1
                elif self.completion_single_decoder_end_to_end:
                    output, completed_scene = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    if not jacc.numel():
                        print('haha')
                        jacc = 0.
                    if not wce.numel():
                        print('haha')
                        wce = 0.
                    mask = (proj_mask - 1) * (-1)  # not on projection mask
                    # l1 = self.l1(completed_scene,  proj_labels_input * mask.unsqueeze(1) + in_vol)
                    if self.edge_loss:
                        l1 = self.l1(completed_scene,
                                     proj_labels_input)
                        edge_mask = None
                        for batch_idx_edge_map in range(proj_labels_mask.shape[0]):
                            # cv2.imwrite('mask_visualization.png', (proj_labels_mask[0]*255).detach().cpu().numpy())
                            edges = cv2.Canny(
                                (proj_labels_mask[batch_idx_edge_map] * 255).detach().cpu().numpy().astype(np.uint8),
                                100, 200)
                            edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
                            # cv2.imwrite('edges_visualization.png', edges)
                            edges_torch = torch.from_numpy(edges / 255 * 5).cuda()
                            if edge_mask is not None:
                                edge_mask = torch.cat((edge_mask, edges_torch.unsqueeze(0)), dim=0)
                            else:
                                edge_mask = edges_torch.unsqueeze(0)
                        l1 += l1 * edge_mask.unsqueeze(1)
                        l1 = l1.mean()
                        # cv2.imwrite('edges_visualization.png', edges)
                    else:
                        l1 = self.l1(completed_scene.permute(0, 2, 3, 1)[mask.bool()],
                                     proj_labels_input.permute(0, 2, 3, 1)[mask.bool()])
                    loss = wce + jacc + l1
                elif self.completion_single_decoder_stage_two:
                    output = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.completion_single_decoder_mask_prediction_stage_two:
                    output = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.completion_stage_two:
                    output = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.completion_stage_two_filtered:
                    output, completed_scene, uf_mask = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.pncnn:
                    # tensor(-1.6432)
                    # tensor(-6.2917)
                    # tensor(-7.0149)
                    # tensor(-4.9338)
                    # tensor(-0.6880)
                    # tensor(2.1007)
                    # tensor(6.1953)
                    # tensor(6.9537)
                    # tensor(12.7295)
                    # tensor(0.3312)
                    in_vol[:,0] += 1.6433
                    in_vol[:,1] += 6.2918
                    in_vol[:,2] += 7.0150
                    in_vol[:,3] += 4.9339
                    in_vol[:,4] += 0.6881
                    proj_labels_input[:,0] += 1.6433
                    proj_labels_input[:,1] += 6.2918
                    proj_labels_input[:,2] += 7.0150
                    proj_labels_input[:,3] += 4.9339
                    proj_labels_input[:,4] += 0.6881
                    in_vol[:,0] /= 1.6433 + 2.1008
                    in_vol[:,1] /= 6.2918 + 6.1954
                    in_vol[:,2] /= 7.0150 + 6.9538
                    in_vol[:,3] /= 4.9339 + 12.7296
                    in_vol[:,4] /= 0.6881 + 0.3313
                    proj_labels_input[:,0] /= 1.6433 + 2.1008
                    proj_labels_input[:,1] /= 6.2918 + 6.1954
                    proj_labels_input[:,2] /= 7.0150 + 6.9538
                    proj_labels_input[:,3] /= 4.9339 + 12.7296
                    proj_labels_input[:,4] /= 0.6881 + 0.3313
                    in_vol *= proj_mask.unsqueeze(1)
                    proj_labels_input *= proj_labels_mask.unsqueeze(1)
                    output = model(in_vol)
                    loss = criterion(output, proj_labels_input, proj_labels_mask)
                    in_vol[:,0] *= 1.6433 + 2.1008
                    in_vol[:,1] *= 6.2918 + 6.1954
                    in_vol[:,2] *= 7.0150 + 6.9538
                    in_vol[:,3] *= 4.9339 + 12.7296
                    in_vol[:,4] *= 0.6881 + 0.3313
                    proj_labels_input[:,0] *= 1.6433 + 2.1008
                    proj_labels_input[:,1] *= 6.2918 + 6.1954
                    proj_labels_input[:,2] *= 7.0150 + 6.9538
                    proj_labels_input[:,3] *= 4.9339 + 12.7296
                    proj_labels_input[:,4] *= 0.6881 + 0.3313
                    output[:,0] *= 1.6433 + 2.1008
                    output[:,1] *= 6.2918 + 6.1954
                    output[:,2] *= 7.0150 + 6.9538
                    output[:,3] *= 4.9339 + 12.7296
                    output[:,4] *= 0.6881 + 0.3313
                    in_vol[:,0] -= 1.6433
                    in_vol[:,1] -= 6.2918
                    in_vol[:,2] -= 7.0150
                    in_vol[:,3] -= 4.9339
                    in_vol[:,4] -= 0.6881
                    output[:,0] -= 1.6433
                    output[:,1] -= 6.2918
                    output[:,2] -= 7.0150
                    output[:,3] -= 4.9339
                    output[:,4] -= 0.6881
                    proj_labels_input[:,0] -= 1.6433
                    proj_labels_input[:,1] -= 6.2918
                    proj_labels_input[:,2] -= 7.0150
                    proj_labels_input[:,3] -= 4.9339
                    proj_labels_input[:,4] -= 0.6881
                    in_vol *= proj_mask.unsqueeze(1)
                    proj_labels_input *= proj_labels_mask.unsqueeze(1)

                    jacc = 0.
                    wce = 0.
                elif self.pncnn_stage_two:
                    completed_scene = None
                    for batch_idx in range(len(path_seq)):
                        scan_file = os.path.join('/home/user/Documents/carla_data/pncnn/pncnn_output',
                                                 path_seq[batch_idx], 'kitti_velodyne',
                                                 path_name[batch_idx].replace('bin', 'npy'))
                        one_completed_scene = np.load(scan_file)
                        one_completed_scene = torch.from_numpy(one_completed_scene).clone().cuda()
                        if completed_scene is not None:
                            completed_scene = torch.cat((completed_scene, one_completed_scene), dim=0)
                        else:
                            completed_scene = one_completed_scene
                    mask = (proj_mask - 1) * (-1)
                    completed_scene = completed_scene * mask.unsqueeze(1) + in_vol
                    output = model(in_vol)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.ipbasic:
                    completed_scene = None
                    for batch_idx in range(len(path_seq)):
                        scan_file = os.path.join('/home/user/Documents/carla_data/ip_basic/demos/outputs',
                                                 path_seq[batch_idx], 'kitti_velodyne',
                                                 path_name[batch_idx].replace('bin', 'npy'))
                        one_completed_scene = np.load(scan_file)
                        one_completed_scene = torch.from_numpy(one_completed_scene).clone().cuda()
                        if completed_scene is not None:
                            completed_scene = torch.cat((completed_scene, one_completed_scene.unsqueeze(0)), dim=0)
                        else:
                            completed_scene = one_completed_scene.unsqueeze(0)
                    mask = (proj_mask - 1) * (-1)
                    completed_scene = completed_scene * mask.unsqueeze(1) + in_vol
                    output = model(in_vol)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.bilateral_filtering:
                    completed_scene = None
                    for batch_idx in range(len(path_seq)):
                        one_completed_scene = in_vol[batch_idx].detach().cpu().numpy()
                        one_completed_scene = cv2.normalize(one_completed_scene, None, 0, 1.0, cv2.NORM_MINMAX,
                                                            dtype=cv2.CV_32F)
                        # one_completed_scene_min = one_completed_scene.min()
                        # one_completed_scene_max = one_completed_scene.max()
                        # one_completed_scene -= one_completed_scene_min
                        # one_completed_scene /= one_completed_scene_max + one_completed_scene_max
                        one_completed_scene_completed = None
                        for image_idx in range(5):
                            one_completed_scene_image = cv2.bilateralFilter(one_completed_scene[image_idx], 5, 25, 25)
                            one_completed_scene_image = cv2.normalize(one_completed_scene_image, None, 0, 1.0,
                                                                      cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            one_completed_scene_image = torch.from_numpy(one_completed_scene_image).clone().cuda()
                            one_completed_scene_image = one_completed_scene_image.unsqueeze(0)
                            if one_completed_scene_completed is not None:
                                one_completed_scene_completed = torch.cat(
                                    (one_completed_scene_completed, one_completed_scene_image), dim=0)
                            else:
                                one_completed_scene_completed = one_completed_scene_image
                        if completed_scene is not None:
                            completed_scene = torch.cat((completed_scene, one_completed_scene_completed.unsqueeze(0)),
                                                        dim=0)
                        else:
                            completed_scene = one_completed_scene_completed.unsqueeze(0)
                    output = model(completed_scene)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.interpolate:
                    output, completed_scene = model(in_vol, proj_mask)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                elif self.full_res:
                    output = model(proj_labels_input)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc
                else:
                    output = model(in_vol)
                    log_out = torch.log(output.clamp(min=1e-8))
                    jacc = self.ls(output, proj_labels)
                    wce = criterion(log_out, proj_labels)
                    loss = wce + jacc

                losses.update(loss.mean().item(), in_vol.size(0))
                if self.completion or self.completion_single_decoder or self.completion_single_decoder_mask_prediction or self.completion_single_decoder_stage_two_perception_loss or self.completion_single_decoder_end_to_end:
                    l1s.update(l1.mean().item(), in_vol.size(0))

                if not (self.completion_single_decoder or self.pncnn or self.completion_single_decoder_mask_prediction):
                    # measure accuracy and record loss
                    argmax = output.argmax(dim=1)
                    evaluator.addBatch(argmax, proj_labels)
                    jaccs.update(jacc.mean().item(),in_vol.size(0))
                    wces.update(wce.mean().item(), in_vol.size(0))


                if save_scans:
                    # get the first scan in batch and project points

                    input_depth_np = in_vol[0][0].cpu().numpy()
                    input_mask_np = proj_mask[0].cpu().numpy()
                    mask_np = proj_labels_mask[0].cpu().numpy()
                    depth_np = proj_labels_input[0][0].cpu().numpy()
                    if self.completion or self.completion_single_decoder or self.pncnn_stage_two or self.ipbasic or self.bilateral_filtering or self.interpolate or self.completion_stage_two_filtered or self.attn or self.completion_single_decoder_mask_prediction or self.completion_single_decoder_end_to_end:
                        completed_scene_np = completed_scene[0][0].cpu().numpy()
                    else:
                        completed_scene_np = np.zeros_like(depth_np)
                    gt_np = proj_labels[0].cpu().numpy()
                    if self.completion_single_decoder or self.pncnn or self.completion_single_decoder_mask_prediction:
                        pred_np = np.zeros_like((gt_np))
                    else:
                        pred_np = argmax[0].cpu().numpy()
                    if self.completion_stage_two_filtered:
                        unfold_mask = uf_mask[0].cpu().numpy()
                    else:
                         unfold_mask = None
                    out = Trainer.make_log_img(input_depth_np,
                                               input_mask_np,
                                               completed_scene_np,
                                               depth_np,
                                               mask_np,
                                               pred_np,
                                               gt_np,
                                               color_fn,
                                               unfold_mask)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            if not (self.completion_single_decoder or self.pncnn or self.completion_single_decoder_mask_prediction):
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
                acc.update(accuracy.item(), in_vol.size(0))
                iou.update(jaccard.item(), in_vol.size(0))
            if self.uncertainty:
                print('Validation set:\n'       
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Hetero avg {hetero.avg}:.4f\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     hetero=hetero_l,
                                                     acc=acc, iou=iou))

                save_to_log(self.log, 'log.txt', 'Validation set:\n'
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Hetero avg {hetero.avg}:.4f\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     hetero=hetero_l,
                                                     acc=acc, iou=iou))
                # print also classwise
                for i, jacc in enumerate(class_jaccard):
                    print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    self.info["valid_classes/"+class_func(i)] = jacc
            else:

                print('Validation set:\n'
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'L1 avg {l1ss.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     l1ss=l1s,
                                                     wces=wces,
                                                     acc=acc, iou=iou))

                save_to_log(self.log, 'log.txt', 'Validation set:\n'
                                                 'Time avg per batch {batch_time.avg:.3f}\n'
                                                 'Loss avg {loss.avg:.4f}\n'
                                                 'Jaccard avg {jac.avg:.4f}\n'
                                                 'L1 avg {l1ss.avg:.4f}\n'
                                                 'WCE avg {wces.avg:.4f}\n'
                                                 'Acc avg {acc.avg:.3f}\n'
                                                 'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                                                loss=losses,
                                                                                jac=jaccs,
                                                                                l1ss=l1s,
                                                                                wces=wces,
                                                                                acc=acc, iou=iou))

                if not (self.completion_single_decoder or self.pncnn or self.completion_single_decoder_mask_prediction):
                    # print also classwise
                    for i, jacc in enumerate(class_jaccard):
                        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                            i=i, class_str=class_func(i), jacc=jacc))
                        save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                            i=i, class_str=class_func(i), jacc=jacc))
                        self.info["valid_classes/" + class_func(i)] = jacc


        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg
