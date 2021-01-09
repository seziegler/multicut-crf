import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys

from src.models.crf_multicut import CrfMulticut
from src.models.mnist_models import SiameseNet, EmbeddingNet, FullNet
from src.utils.mnist import fit, check_gpu, get_mnist, get_dataloaders
from src.evaluation.metrics import NMI, AdjustedRand, BestMap
from src.cycle_monitoring.visualizations import visualize_cycles



if __name__ == '__main__':

    orig_stdout = sys.stdout
    
    device = check_gpu()
    print('Device used: ', device)
    
    nb_images_in_batch = 10
    
    nb_batches_train = 1000
    nb_batches_val = 100
    
    lr = 0.0001
    n_epochs = 20
    log_interval = 100
    
    # get MNIST data, set download to True if you don't have the data yet
    train_dataset, test_dataset = get_mnist(download=False)
    
    # get torch dataloaders that return fully connected batches
    siamese_train_loader, siamese_test_loader = get_dataloaders(train_dataset,test_dataset,nb_images_in_batch,nb_batches_train,nb_batches_val)
    
    ########################
    # only frontend
    print('Training model without CRF')
    
    
    sys.stdout = open('./MNIST/training_no_crf.txt', 'w')
    
    model = FullNet(CrfMulticut(5, True, True), SiameseNet(EmbeddingNet(), True, True), device, device, only_frontend=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    weights = [0.9,0.1]
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    
    # fit the model
    cycle_counts_rounded, cycle_counts, val_cycle_counts_rounded, val_cycle_counts = fit(siamese_train_loader, siamese_test_loader, 
                                                                                            model, loss_fn, optimizer, 
                                                                                            scheduler, n_epochs, log_interval, 
                                                                                            metrics =[#AccumulatedAccuracyMetric(),
                                                                                                      NMI(),
                                                                                                      AdjustedRand(),
                                                                                                      BestMap()],
                                                                                            exp=False)
    
    # show cycles during training on training set
    visualize_cycles(cycle_counts_rounded, cycle_counts,n_epochs,save=True,file='./MNIST/only_frontend_cycles_train.png')
    # show cycles during training on validation set
    visualize_cycles(val_cycle_counts_rounded, val_cycle_counts,n_epochs,save=True,file='./MNIST/only_frontend_cycles_val.png')
    
    sys.stdout = orig_stdout
    
    #########################################################################
    # Combined model with SiameseNet as frontend and CrfMulticut on top, with increasing exponents
    print('Training model with CRF (with increasing exponent)')
    
    sys.stdout = open('./MNIST/training_with_crf_exp.txt', 'w')
    
    model = FullNet(CrfMulticut(5, True, True), SiameseNet(EmbeddingNet()), frontend_device=device, crf_device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    weights = [0.9,0.1]
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    #scheduler = None
    #n_epochs = 5
    #log_interval = log_interval
    
    # fit the model
    cycle_counts_rounded, cycle_counts, val_cycle_counts_rounded, val_cycle_counts = fit(siamese_train_loader, siamese_test_loader, 
                                                                                            model, loss_fn, optimizer, 
                                                                                            scheduler, n_epochs, log_interval, 
                                                                                            metrics =[#AccumulatedAccuracyMetric(),
                                                                                                      NMI(),
                                                                                                      AdjustedRand(),
                                                                                                      BestMap()],
                                                                                            exp=True)
    
    # show cycles during training on training set
    visualize_cycles(cycle_counts_rounded, cycle_counts,n_epochs,save=True,file='./MNIST/crf_cycles_train_exp.png')
    # show cycles during training on validation set
    visualize_cycles(val_cycle_counts_rounded, val_cycle_counts,n_epochs,save=True,file='./MNIST/crf_cycles_val_exp.png')
    
    sys.stdout = orig_stdout
    
    
    ##########################
    # Combined model with SiameseNet as frontend and CrfMulticut on top
    print('Training model with default CRF (no exponent)')
    
    sys.stdout = open('./MNIST/training_with_crf_default.txt', 'w')
    
    model = FullNet(CrfMulticut(5, True, True, 1), SiameseNet(EmbeddingNet()), frontend_device=device, crf_device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    weights = [0.9,0.1]
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    #scheduler = None
    #n_epochs = 5
    #log_interval = log_interval
    
    # fit the model
    cycle_counts_rounded, cycle_counts, val_cycle_counts_rounded, val_cycle_counts = fit(siamese_train_loader, siamese_test_loader, 
                                                                                            model, loss_fn, optimizer, 
                                                                                            scheduler, n_epochs, log_interval, 
                                                                                            metrics =[#AccumulatedAccuracyMetric(),
                                                                                                      NMI(),
                                                                                                      AdjustedRand(),
                                                                                                      BestMap()],
                                                                                            exp=False)
    
    # show cycles during training on training set
    visualize_cycles(cycle_counts_rounded, cycle_counts,n_epochs,save=True,file='./MNIST/crf_cycles_train_default.png')
    # show cycles during training on validation set
    visualize_cycles(val_cycle_counts_rounded, val_cycle_counts,n_epochs,save=True,file='./MNIST/crf_cycles_val_default.png')
    
    sys.stdout = orig_stdout
    print('Done!')




