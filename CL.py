import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

class Classifier(nn.Module):
    def __init__(self, in_fea, hid_fea, out_fea, drop_out=0.5):
        super(Classifier, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, out_fea))

    def forward(self, doc_fea):
        z = F.normalize(self.projector(doc_fea),dim=1)
        return z


class UCL(nn.Module):
    def __init__(self, in_fea, out_fea, temperature=0.5):
        super(UCL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, out_fea),
            nn.BatchNorm1d(out_fea),
            nn.ReLU(inplace=True),
            nn.Linear(out_fea, out_fea))
        self.tem = temperature
    
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def forward(self, doc_fea):
        out = self.projector(doc_fea)
        out = F.normalize(out, dim=1)
        dim = out.shape[0]
        odd, even = [i for i in range(dim) if i % 2 == 0], [i for i in range(dim) if i % 2 != 0]
        out_1, out_2 = out[even], out[odd]
        
        batch_size = 2560
        l1 = self.batch_loss(out_1, out_2, batch_size)
        l2 = self.batch_loss(out_2, out_1, batch_size)
        
        loss = (l1+l2) * 0.5
        return loss.mean()
    
    def batch_loss(self, out_1, out_2, batch_size):
        device = out_1.device
        num_nodes = out_1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tem)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(out_1[mask], out_1))  # [B, N]
            between_sim = f(self.sim(out_1[mask], out_2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

        
class WCL(nn.Module):
    def __init__(self, in_fea, hid_fea, temperature=0.5):
        super(WCL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, hid_fea))
        self.projector_2 = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, hid_fea))
        self.tem = temperature
   
    @torch.no_grad()
    def build_connected_component(self, dist):
        b = dist.size(0)
        dist = dist.fill_diagonal_(-2)
        device = dist.device
        x = torch.arange(b).unsqueeze(1).repeat(1,1).flatten().to(device)
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])
        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels).to(device)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * dignal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, doc_fea, share=True):
        out = self.projector(doc_fea)
        device = out.device
        out = F.normalize(out, dim=1)
        dim = out.shape[0]
        odd, even = [i for i in range(dim) if i % 2 == 0], [i for i in range(dim) if i % 2 != 0]
        out_1, out_2 = out[even], out[odd]
        b = out_1.shape[0]
        

        if share:
            mask1 = self.build_connected_component(out_1 @ out_1.T).float()
            mask2 = self.build_connected_component(out_2 @ out_2.T).float()

        else:
            out = self.projector_2(doc_fea)
            out = F.normalize(out, dim=1)
            out_1, out_2 = out[even], out[odd]
            mask1 = self.build_connected_component(out_1 @ out_1.T).float()
            mask2 = self.build_connected_component(out_2 @ out_2.T).float()
        diagnal_mask = torch.eye(b, b).to(device)
        graph_loss = self.sup_contra(out_1 @ out_1.T / self.tem, mask2, diagnal_mask)
        graph_loss += self.sup_contra(out_2 @ out_2.T / self.tem, mask1, diagnal_mask)
        graph_loss /= 2
        return graph_loss

    