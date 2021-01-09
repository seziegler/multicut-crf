import itertools
import os
import subprocess
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.utils.crf import get_graph_idx

# functions for the MNIST experiment


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, log_interval, metrics=[], exp=True):

    cycle_counts_rounded_list = []
    cycle_inequality_counts_list = []
    val_cycle_counts_rounded_list = []
    val_cycle_counts_list = []

    for epoch in range(0, n_epochs):

        print('Epoch {}/{} '.format(epoch+1,n_epochs))


        # Training Set
        train_loss, metrics_dict, outputs, cycle_counts_rounded, cycle_inequality_counts = train_epoch(train_loader, model, loss_fn, optimizer, log_interval, metrics)

        cycle_counts_rounded_list.append(cycle_counts_rounded)
        cycle_inequality_counts_list.append(cycle_inequality_counts)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\n{}: {}'.format(metric.name(), np.mean(metrics_dict[metric.name()]))

        # Validation Set
        val_loss, metrics_dict, outputs, val_cycle_counts_rounded, val_cycle_counts, val_energy = test_epoch(val_loader, model, loss_fn, metrics)

        val_cycle_counts_rounded_list.append(val_cycle_counts_rounded)
        val_cycle_counts_list.append(val_cycle_counts)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\n{}: {}'.format(metric.name(), np.mean(metrics_dict[metric.name()]))

        message += '\nVal_energy: {}'.format(np.mean(val_energy))

        print(message, flush=True)

        if exp:
            vals,invals = zip(*cycle_inequality_counts)
            if np.mean(invals)<10:
                model.multicut.p += 0.1
                print('####################################### CRF exponent updated to',model.multicut.p)

        if scheduler:
            scheduler.step()


    return cycle_counts_rounded_list, cycle_inequality_counts_list, val_cycle_counts_rounded_list, val_cycle_counts_list


def train_epoch(train_loader, model, loss_fn, optimizer, log_interval, metrics):
    print('######## Start of Epoch #########', flush=True)
    # init dict for metrics with name as key and empty list for values
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric.name()] = []

    model.train()
    losses = []
    #total_loss = 0

    cycle_counts_rounded_list = []
    cycle_counts_list = []

    frontend_losses = []
    RNN_losses = []
    energy_values = []


    for batch_idx, (data,target,indices_list,labels,edge_index,j_edge_indices,cliques_ints,edges) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs, cycle_counts_rounded, cycle_counts, unaries = model(*data,edge_index,j_edge_indices,cliques_ints)

        # check device type
        dev = outputs.get_device()
        if dev == -1:
            dev = 'cpu'

        #outputs = outputs.to('cpu')
        target = target.to(dev)

        cycle_counts_rounded_list.append(cycle_counts_rounded)
        cycle_counts_list.append(cycle_counts)


        combined_losses = []
        for out in (unaries, outputs):
            # out is None in one case if only frontend model is trained
            if out!=None:

                loss_outputs = loss_fn(out,target)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

                if unaries!=None:
                    combined_losses.append(loss)

        if unaries!=None:

            frontend_losses.append(combined_losses[0].clone().detach())
            RNN_losses.append(combined_losses[1].clone().detach())


            # add individual losses
            loss = sum(combined_losses)

            # alternative: weight losses
            #loss = 0.9*combined_losses[0] + 0.1*combined_losses[1]
            # alternative: only output loss
            #loss = combined_losses[1]

        # solve Multicut to obtain final clusters
        join_probs = outputs.clone().detach()[:,0] # 0 join, 1 cut
        pred_labels, energy = solve_multicut(indices_list, edges, join_probs)
        #print('Actual Labels: ', labels)
        #print('Predicted Labels: ', pred_labels)
        energy_values.append(energy)

        losses.append(loss.item())
        #total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metrics_dict[metric.name()].append(metric(pred_labels,labels))

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(
                  batch_idx, len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                  message += '\n{}: {}'.format(metric.name(), np.mean(metrics_dict[metric.name()]))

            if unaries!=None:
                message += '\nFrontend Loss: {}'.format(torch.mean(torch.stack(frontend_losses)))
                message += '\nCRF Loss: {}'.format(torch.mean(torch.stack(RNN_losses)))

            message += '\nEnergy: {}'.format(np.mean(energy_values))

            print(message)
            #losses = []



    #total_loss /= (batch_idx + 1)
    total_loss = np.mean(losses)

    #################################
    print('################### End of Epoch #################',flush=True)
    if unaries!=None:
        print('Frontend Loss: ', torch.mean(torch.stack(frontend_losses)))
        print('CRF Loss: ', torch.mean(torch.stack(RNN_losses)))
    vals_r,invals_r = zip(*cycle_counts_rounded_list)
    vals,invals = zip(*cycle_counts_list)
    print('Avg number of invalid cycles (rounded): ',np.mean(invals_r))
    print('Avg number of invalid cycle inequalities: ',np.mean(invals))
    print('Train_Energy: ',np.mean(energy_values))
    ##################################

    return total_loss, metrics_dict, outputs, cycle_counts_rounded_list, cycle_counts_list


def test_epoch(val_loader, model, loss_fn, metrics):
    with torch.no_grad():
        # init dict for metrics with name as key and empty list for values
        metrics_dict = {}
        for metric in metrics:
            metrics_dict[metric.name()] = []

        model.eval()
        val_loss = 0

        val_cycle_counts_rounded_list = []
        val_cycle_counts_list = []

        energy_values = []

        for batch_idx, (data,target,indices_list,labels,edge_index,j_edge_indices,cliques_ints,edges) in enumerate(val_loader):
            target = target if len(target) > 0 else None

            outputs, val_cycle_counts_rounded, val_cycle_counts, unaries = model(*data,edge_index,j_edge_indices,cliques_ints)

            # check device type
            dev = outputs.get_device()
            if dev == -1:
                dev = 'cpu'

            #outputs = outputs.to('cpu')
            target = target.to(dev)

            val_cycle_counts_rounded_list.append(val_cycle_counts_rounded)
            val_cycle_counts_list.append(val_cycle_counts)

            combined_losses = []
            for out in (unaries, outputs):
                # out is None in the case when only frontend model is trained
                if out!=None:

                    loss_outputs = loss_fn(out,target)
                    loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

                    if unaries!=None:
                        combined_losses.append(loss)

            if unaries!=None:
                # add individual losses
                loss = sum(combined_losses)
                # alternative: weight losses
                #loss = 0.9*combined_losses[0] + 0.1*combined_losses[1]
                # alternative: only output loss
                #loss = combined_losses[1]


            val_loss += loss.item()

            #print('Start solving Validation')
            join_probs = outputs.clone().detach()[:,0]
            pred_labels, energy = solve_multicut(indices_list, edges, join_probs)
            #print('Actual Labels Validation: ', labels)
            #print('Predicted Labels Validation: ', pred_labels)
            energy_values.append(energy)


            for metric in metrics:
                metrics_dict[metric.name()].append(metric(pred_labels,labels))

    val_loss /= len(val_loader)
    return val_loss, metrics_dict, outputs, val_cycle_counts_rounded_list, val_cycle_counts_list, energy_values


def check_gpu():

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device


def solve_multicut(nodes, edges, outputs):

    outputs = outputs.to('cpu')

    nb_nodes = len(nodes)
    nb_edges = len(edges)

    # adapt indices for multicut library
    new_idx = np.arange(0,nb_nodes,1)
    edges_new_idx = list(itertools.combinations(new_idx,2))

    # combine edges and outputs
    combined = np.column_stack((np.array(edges_new_idx),outputs))

    # create txt file from data
    filename="./MNIST/solve_mc_text_input.txt"
    np.savetxt(filename,combined, fmt='%d %d %.10f',header=str(nb_nodes)+' '+str(nb_edges),comments='')

    file_out="../mc_result.h5"
    # solve multicut problem
    subprocess.call(['./graph_lib/graph/build/solve-regular', '-i', filename, '-o', file_out],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # load result
    f = h5py.File(file_out, 'r')
    labels = np.array(f['labels'])
    energy = np.array(f['energy-value'])

    # clean up files
    os.remove(filename)
    os.remove(file_out)

    return labels, energy

############
### Data ###
############

class SiameseMNIST_random_fc_batches(Dataset):

    def __init__(self, mnist_dataset, nb_images_in_batch, nb_batches):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        self.train_labels = self.mnist_dataset.train_labels
        self.train_data = self.mnist_dataset.train_data
        self.labels_set = set(self.train_labels.numpy())
        self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                 for label in self.labels_set}

        self.nb_images = nb_images_in_batch
        self.nb_batches = nb_batches
        assert(len(self.train_data)%self.nb_batches==0)
        #self.indices_tensor = torch.zeros(size=[self.nb_batches,self.nb_images],dtype=int)
        shuffle_idx = torch.randperm(len(self.train_data))
        self.indices_tensor = shuffle_idx.view(-1,self.nb_images)[:self.nb_batches]


    def __getitem__(self, index):
        indices_list = self.indices_tensor[index]
        cliques = torch.combinations(indices_list, 3)
        edges = torch.combinations(indices_list, 2)

        edge_index, j_edge_indices, cliques_ints = get_graph_idx(edges, cliques)

        # save labels
        labels = [self.train_labels[i].item() for i in indices_list]

        # classification target for Siamese Net
        targets = []
        imgs1 = []
        imgs2 = []
        for index1, index2 in edges:
            img1, label1 = self.train_data[index1], self.train_labels[index1].item()
            img2, label2 = self.train_data[index2], self.train_labels[index2].item()

            img1 = Image.fromarray(img1.numpy(), mode='L')
            img2 = Image.fromarray(img2.numpy(), mode='L')
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs1.append(img1.unsqueeze(0))
            imgs2.append(img2.unsqueeze(0))
            target = 0 if label1==label2 else 1 #0=join, 1=cut
            targets.append(target)

        targets = torch.tensor(targets)

        return (torch.stack(imgs1).squeeze(),torch.stack(imgs2).squeeze()) , targets, indices_list, labels, edge_index, j_edge_indices, cliques_ints, edges

    def __len__(self):
        return self.nb_batches


def custom_mnist_collate(batch):
    imgs,targets,indices_list,labels,edge_index,j_edge_indices,cliques_ints,edges = zip(*batch)
    imgs1, imgs2 = zip(*imgs)
    indices_list = indices_list[0]
    labels = labels[0]
    edge_index = edge_index[0]
    j_edge_indices = j_edge_indices[0]
    cliques_ints = cliques_ints[0]
    edges = edges[0]

    return (torch.cat(imgs1).unsqueeze(0).permute(1,0,2,3),torch.cat(imgs2).unsqueeze(0).permute(1,0,2,3)),torch.cat(targets),indices_list,labels,edge_index,j_edge_indices,cliques_ints,edges


def get_mnist(download=False):

    mean, std = 0.1307, 0.3081

    train_dataset = MNIST('./MNIST', train=True, download=download,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))
    test_dataset = MNIST('./MNIST', train=False, download=download,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))

    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, nb_images_in_batch, nb_batches_train, nb_batches_val):

    siamese_train_dataset = SiameseMNIST_random_fc_batches(train_dataset, nb_images_in_batch, nb_batches_train)
    siamese_test_dataset = SiameseMNIST_random_fc_batches(test_dataset, nb_images_in_batch, nb_batches_val)

    # has to be 1, as dataloader already contains specified batches, if you want to increase the number of images being processed per batch increase nb_images_in_batch
    batch_size = 1

    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_mnist_collate)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_mnist_collate)

    return siamese_train_loader, siamese_test_loader