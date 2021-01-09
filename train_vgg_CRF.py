import os
import sys
import pickle
import torch
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath

from rcf_vgg.data_loader import BSDS_RCFLoader
from rcf_vgg.models import RCF
from rcf_vgg.functions import cross_entropy_loss_RCF
from rcf_vgg.utils import Logger, save_checkpoint

from src.models.crf_multicut import CrfMulticut
from src.models.rcf_crf import RCF_CRFasRNN
from src.utils.rcf import get_model_params, get_opt, get_scheduler
from src.utils.rcf_crf import train


if __name__ == '__main__':

    print('Inits model and loads data...')

    ### Init Model ###

    # init RCF

    rcf_model = RCF()
    rcf_model.cuda()

    # load pretrained model
    pretrained_dict = torch.load('./rcf_vgg/tmp/RCF_stepsize3/checkpoint_epoch8.pth')#,map_location=torch.device('cpu'))
    rcf_model.load_state_dict(pretrained_dict['state_dict'])

    # init crf
    crf_model = CrfMulticut(5, True, True, 1)

    model = RCF_CRFasRNN(rcf_model, crf_model, only_rcf=False, only_edgemap=False, rcf_backbone='VGG')
    model = model.cuda()

    ### Args ###

    batch_size=1
    lr = 1e-6
    momentum = 0.9
    weight_decay = 2e-4
    stepsize = 3
    gamma = 0.1
    start_epoch = 0
    maxepoch = 30
    itersize = 10
    print_freq = 100
    tmp = 'SGDs/vgg_WS_default'

    # Prepare training
    net_parameters_id = get_model_params(model)
    optimizer = get_opt(lr, momentum, weight_decay, net_parameters_id)
    scheduler = get_scheduler(optimizer, stepsize, gamma)  


    # load data

    train_dataset = BSDS_RCFLoader(split="train")

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
            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch),shape_dict=shape_dict,contour_cue_dict=contour_cue_dict, itersize=itersize, print_freq=print_freq, maxepoch=maxepoch, backbone='VGG', method='power')
        '''test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record' % epoch))'''
        log.flush()  # write log
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