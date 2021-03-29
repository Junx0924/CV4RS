import numpy as np, torch

# implemented according to paper: https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf
class Distance():
    def __init__(self, lower_cutoff =0.5,upper_cutoff =1.4):
        self.lower_cutoff =  lower_cutoff
        self.upper_cutoff =  upper_cutoff

    def __call__(self, batch, labels):
        """ Generate tripets based on category labels

        Args:
            batch (tensor):  dim [batchsize,embedding dim]
            labels (tensor): category labels, dim [batchsize,1]

        Returns:
            tuple: index [anchor, positive, negative]
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]
        distances = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)

        anchors, positives, negatives = [],[],[]

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]

            if np.sum(pos)>1:
                anchors.append(i)
                q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
                #Sample positives randomly
                pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))
                #Sample negatives by distance
                negatives.append(np.random.choice(bs,p=q_d_inv))

        return anchors, positives, negatives


    def inverse_sphere_distances(self, batch, anchor_to_all_dists, labels, anchor_label):
            dists        = anchor_to_all_dists
            bs,dim       = len(dists),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            # torch.clamp(_,min=1e-8) for stability, doesn't work well on recall@1
            #log_q_d_inv = ((2.0 - float(dim)) * torch.log(dists) - (float(dim-3) / 2) * torch.log(torch.clamp(1.0 - 0.25*(dists*dists), min=1e-8)))
            # make sure the embedings are normalized
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dists) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dists.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
            # errors where there are no available negatives (for high samples_per_class cases).
            # q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()


    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.sqrt()
