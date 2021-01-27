import torch, torch.nn as nn
from tqdm import tqdm
import numpy as np

"""================================================================================================="""
class Fast_moco(torch.nn.Module):
    def __init__(self, config,moco_temperature = 0.07, moco_momentum= 0.9, moco_n_key_batches= 30,moco_temp_lr= 0.0005,lower_cutoff=0.5, upper_cutoff=1.4):
        
        super(Fast_moco, self).__init__()

        self.temperature   = moco_temperature
        self.momentum      = moco_momentum
        self.n_key_batches = moco_n_key_batches
        self.lr          = moco_temp_lr
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff
        batch_size = config['dataloader']['batch_size']
        self.reference_labels = torch.zeros(batch_size).to(torch.long).to(config['device'])

        self.diva_features = config['diva_features'] 
        self.sz_embedding = config['sz_embedding']
    def update_memory_queue(self, embeddings):
        self.memory_queue = self.memory_queue[len(embeddings):,:]
        self.memory_queue = torch.cat([self.memory_queue, embeddings], dim=0)

    def create_memory_queue(self, model, dataloader, device, opt_key=None):
        with torch.no_grad():
            _ = model.eval()
            _ = model.to(device)

            self.memory_queue = []
            counter = 0
            load_count  = 0
            total_count = self.n_key_batches//len(dataloader) + int(self.n_key_batches%len(dataloader)!=0)
            while counter<self.n_key_batches-1:
                load_count += 1
                for i,input_tuple in enumerate(tqdm(dataloader, 'Filling memory queue [{}/{}]...'.format(load_count, total_count), total=len(dataloader))):
                    embed = model(input_tuple[0].type(torch.FloatTensor).to(device))

                    if opt_key is not None:
                        embed = embed.split(self.sz_embedding // len(self.diva_features), dim = 1)
                        ind = self.diva_features.index(opt_key)
                        embed = embed[ind].cpu()
                    else:
                        embed = embed.cpu()

                    self.memory_queue.append(embed)

                    counter+=1
                    if counter>=self.n_key_batches:
                        break

            self.memory_queue = torch.cat(self.memory_queue, dim=0).to(device)

        self.n_keys = len(self.memory_queue)

    def shuffleBN(self, bs):
        forward_inds  = torch.randperm(bs).long().cuda()
        backward_inds = torch.zeros(bs).long().cuda()
        value = torch.arange(bs).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds


    def forward(self, query_batch, key_batch):
        """
        Args:
            query_batch: embeddings from images, torch.Tensor (BS x DIM)
            key_batch: embeddings from augumented images, torch.Tensor(BS x DIM)
        """
        bs  = len(query_batch)

        l_pos = query_batch.view(bs, 1, -1).bmm(key_batch.view(bs, -1, 1)).squeeze(-1)
        l_neg = query_batch.view(bs, -1).mm(self.memory_queue.T)

        ### Compute Distance Matrix
        bs,dim  = len(query_batch),query_batch.shape[-1]

        ab = torch.mm(query_batch, self.memory_queue.T).detach()
        a2 = torch.nn.CosineSimilarity()(query_batch, query_batch).unsqueeze(1).expand_as(ab).detach()
        b2 = torch.nn.CosineSimilarity()(self.memory_queue, self.memory_queue).unsqueeze(0).expand_as(ab).detach()
        #Euclidean Distances
        distance_weighting = (-2.*ab+a2+b2).clamp(min=0).sqrt()
        distances = (-2.*ab+a2+b2).clamp(min=0).sqrt().clamp(min=self.lower_cutoff)

        #Likelihood Weighting
        distance_weighting = ((2.0 - float(dim)) * torch.log(distances) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (distances.pow(2))))
        distance_weighting = torch.exp(distance_weighting - torch.max(distance_weighting))
        distance_weighting[distances>self.upper_cutoff] = 0
        distance_weighting = distance_weighting.clamp(min=1e-45)
        distance_weighting = distance_weighting/torch.sum(distance_weighting, dim=0)

        ###
        l_neg = l_neg*distance_weighting

        ### INCLUDE SHUFFLE BN
        logits = torch.cat([l_pos, l_neg], dim=1)

        if isinstance(self.temperature, torch.Tensor):
            loss = torch.nn.CrossEntropyLoss()(logits/self.temperature.clamp(min=1e-8, max=1e4), self.reference_labels)
        else:
            loss = torch.nn.CrossEntropyLoss()(logits/self.temperature, self.reference_labels)

        return loss
