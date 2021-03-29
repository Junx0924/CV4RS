import numpy as np, torch


class MultiLabelSemihard():
    """MultiLabel semihard.
       reference: https://github.com/Junx0924/multilabel-deep-metric/blob/master/src/utils.py.
    """    
    def __init__(self,max_negatives_per_pos =3,max_trips_per_anchor =3):
        self.max_negatives_per_pos =  max_negatives_per_pos
        self.max_trips_per_anchor =  max_trips_per_anchor

    def __call__(self, batch, labels):
        """ Generate tripets based on multi-hot labels
        Args:
            batch (tensor):  dim [batchsize,embedding dim]
            labels (tensor): multi-hot labels, dim [batchsize,L]
        Returns:
            tuple: index [anchor, positive, negative]
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach()
        bs = batch.shape[0]
        distances = self.pdist(batch.detach())
        labels_sim = torch.mm(labels,labels.t())

        anchors, positives, negatives = [],[],[]
        for i in range(bs):
            # get similarity scores for multi-hot labels
            sim_scores= labels_sim[i,:]
            # get embedding distances
            dist = distances[i,:]
            dist_order=torch.argsort(dist)
            sim_scores_ordered=sim_scores[dist_order]
            num_anchor_triplets=0
            positive_idcs= torch.where(sim_scores_ordered>0)[0]
            num_fine=0
            num_coarse=0
            # mine positives first, starting with the back
            for pos_idx in torch.flip(positive_idcs,[0]):
                # if it is the anchor
                if dist_order[pos_idx]==i:
                    continue
                pos_sim=sim_scores_ordered[pos_idx]
                # generate fine triplets
                a = sim_scores_ordered[:pos_idx]<pos_sim
                b = sim_scores_ordered[:pos_idx]>0
                positive_misorderings= a & b
                for neg_idx in torch.where(positive_misorderings)[0]:
                    anchors.append(i)
                    positives.append(dist_order[pos_idx])
                    negatives.append(dist_order[neg_idx])
                    num_anchor_triplets+=1
                    num_fine+=1
                # generate coarse triplets
                zero_idcs=torch.where(sim_scores_ordered[:pos_idx]==0)[0]
                if len(zero_idcs)==0:
                    continue
                num_negatives=min(self.max_negatives_per_pos,zero_idcs.shape[0])
                for _ in range(0,num_negatives):
                    # choose a negative randomly, because there are a lot of negatives
                    # and since as we go down the positive_idcs, the previous zero_idcs
                    # is included, so we don't want to keep choosing the same negatives
                    k=np.random.randint(0,len(zero_idcs))
                    neg_idx=zero_idcs[k]
                    anchors.append(i)
                    positives.append(dist_order[pos_idx])
                    negatives.append(dist_order[neg_idx])
                    num_anchor_triplets+=1
                    num_coarse+=1
                if num_anchor_triplets>=self.max_trips_per_anchor:
                    break
        return anchors, positives, negatives

    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.sqrt()
