import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import aggregate
from GCN import GCN

class DELIGHT(nn.Module):
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params):
        super(DELIGHT, self).__init__()
        self.adj = adj_dict
        self.feature = features_dict
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.type_num = len(params.type_num_node)
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb
        self.device = params.device
        self.GCNs = nn.ModuleList()
        self.GCNs_2 = nn.ModuleList()

        for i in range(1, self.type_num):
            self.GCNs.append(GCN(self.in_features_dim[i], self.out_features_dim[i]).to(self.device))
            self.GCNs_2.append(GCN(self.out_features_dim[i], self.out_features_dim[i]).to(self.device))

    def embed_component(self, norm=True):
        output = []
        for i in range(self.type_num - 1):
            
            if i == 1 and self.concat_word_emb:
                temp_emb = torch.cat([
                    F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)),
                                p=self.drop_out, training=self.training), self.feature['word_emb']], dim=-1)
                output.append(temp_emb)     
            elif i == 0:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                            self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],self.feature[str(i + 1)], identity=True)),
                            p=self.drop_out, training=self.training)
                output.append(temp_emb)       
            else:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],self.feature[str(i + 1)])),
                                p=self.drop_out, training=self.training)
                output.append(temp_emb)
        refined_text_input = aggregate(self.adj, output, self.type_num - 1) 
        if norm:
            refined_text_input_normed = []
            for i in range(self.type_num - 1):
                refined_text_input_normed.append(refined_text_input[i] / (refined_text_input[i].norm(p=2, dim=-1,keepdim=True) + 1e-9))
        else:
            refined_text_input_normed = refined_text_input
        return refined_text_input_normed
    
    def forward(self, epoch):
        refined_text_input_normed = self.embed_component()
        
        Doc_features = torch.cat(refined_text_input_normed, dim=-1)
        

        return Doc_features
