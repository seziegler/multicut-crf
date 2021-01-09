import os
import sys
import pickle
import torch
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath

from rcf_resnet import models
from rcf_resnet.data_loader import BSDS_RCFLoader
from rcf_vgg.utils import Logger, save_checkpoint

from src.models.crf_multicut import CrfMulticut
from src.models.rcf_crf import RCF_CRFasRNN
from src.utils.rcf import get_scheduler, cross_entropy_loss_RCF, make_optim
from src.utils.rcf_crf import train


if __name__ == '__main__':

    print('Inits model and loads data...')

    ##################
    ### Init Model ###
    ##################

    # init RCF

    rcf_model = models.resnet101(pretrained=False).cuda()

    # load pretrained model for RCF Resnet Version

    path_to_model_only_rcf = 'rcf_resnet/only-final-lr-0.01-iter-130000.pth'
    checkpoint = torch.load(path_to_model_only_rcf)#,map_location=torch.device('cpu'))
    rcf_model.load_state_dict(checkpoint)

    # init crf

    crf_model = CrfMulticut(5, True, True, p=1)

    # init combined model RCF_CRFasRNN

    model = RCF_CRFasRNN(rcf_model, crf_model, only_rcf=False, only_edgemap=False)
    model = model.cuda()

    ############
    ### Args ###
    ############

    batch_size=1
    lr = 1e-3
    momentum = 0.9
    weight_decay = 2e-4
    stepsize = 3
    gamma = 0.1
    start_epoch = 0
    maxepoch = 20
    itersize = 10
    print_freq = 100
    tmp = 'SGDs/resnet_WS_default'

    # Prepare training
  

    optimizer = make_optim(model, lr, momentum, weight_decay)
    scheduler = get_scheduler(optimizer, stepsize, gamma) 



    # load data

    train_dataset = BSDS_RCFLoader(split="train")
    #test_dataset = BSDS_RCFLoader(split="test")

    train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            num_workers=8, drop_last=True,shuffle=True)


    # load dicts

    shape_dict = pickle.load( open( "dicts/full_line_shape_dict_8.p", "rb" ) )
    contour_cue_dict = pickle.load( open( "dicts/full_line_contour_cue_dict_8.p", "rb" ) )

    print('Everything loaded! Starts training...')

    # log
    THIS_DIR = abspath('')
    TMP_DIR = join(THIS_DIR, tmp)
    if not isdir(TMP_DIR):
        os.makedirs(TMP_DIR)
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('sgd',lr)))

    # print to file or console
    orig_stdout = sys.stdout
    sys.stdout = log
    #sys.stdout = orig_stdout

    # train

    train_loss = []
    train_loss_detail = []
    for epoch in range(start_epoch, maxepoch):
        '''if epoch == 0:
            print("Performing initial testing...")
            multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
                save_dir = join(TMP_DIR, 'initial-testing-record'))'''

        tr_avg_loss, tr_detail_loss, shape_dict, contour_cue_dict = train(
            train_loader, model, cross_entropy_loss_RCF, optimizer, scheduler, epoch,
            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch),shape_dict=shape_dict,contour_cue_dict=contour_cue_dict, itersize=itersize, print_freq=print_freq, maxepoch=maxepoch, backbone='Resnet',method='power')
        '''test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record' % epoch))'''
        log.flush() # write log
        # Save checkpoint
        save_file = os.path.join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'p': model.crf.p
                        }, filename=save_file)
        scheduler.step() # will adjust learning rate
        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss


    sys.stdout = orig_stdout
    print('Finished training!')