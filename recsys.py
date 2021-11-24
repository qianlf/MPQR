

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

class RecSys(nn.Module):

    def __init__(self, embedding_dim
                 , cnn_channel
                 , embeddings):
        super(RecSys, self).__init__()
        self.emb_dim = embedding_dim
        self.out_channel = cnn_channel
        self.embedding_manager = embeddings


        self.convnet1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, self.out_channel, kernel_size=(1, embedding_dim))),
            ('relu1', nn.ReLU())

        ]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(1, self.out_channel, kernel_size=(2, embedding_dim))),
            ('relu2', nn.ReLU())

        ]))

        self.convnet3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(1, self.out_channel, kernel_size=(3, embedding_dim))),
            ('relu3', nn.ReLU())
        ]))

        self.fc1 = nn.Linear(self.out_channel, 1)
        self.fc2 = nn.Linear(self.out_channel, 1)
        self.fc3 = nn.Linear(self.out_channel, 1)
        
        self.fc333 = nn.Linear(embedding_dim, 1)
        
        
        
        
        
        
        self.convnetp1 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(1, self.out_channel, kernel_size=(1, embedding_dim))),
            ('relu4', nn.ReLU())

        ]))

        self.convnetp2 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(1, self.out_channel, kernel_size=(2, embedding_dim))),
            ('relu5', nn.ReLU())

        ]))

        self.convnetp3 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(1, self.out_channel, kernel_size=(3, embedding_dim))),
            ('relu6', nn.ReLU())
        ]))


        
        ones=torch.Tensor(torch.randn(embedding_dim,embedding_dim)) 
        self.w1=torch.nn.Parameter(ones) 
        self.w2=torch.nn.Parameter(ones)
        self.wq=torch.nn.Parameter(ones)
        

        self.fc_new_1 = nn.Linear(6, 1)
        self.fc_new_2 = nn.Linear(self.out_channel, 1)
        
        
        self.fcp_new_1 = nn.Linear(6, 1)
        # batch x 32 x 1 => batch x 1 x 1 (batch x 1)
        self.fcp_new_2 = nn.Linear(self.out_channel, 1)

    def forward(self, rank):
        """
         === Ranking ===
        """
        emb = self.embedding_manager


        emb_rank_r = (emb.ru_embeddings(rank[0]))#+embe.reu_embeddings(rank[0])*0.1
        emb_rank_a =(emb.au_embeddings(rank[1]))#*0.9+(embe.aeu_embeddings(rank[1]))*0.1
        emb_rank_acc = (emb.au_embeddings(rank[2]))
        rank_q, rank_q_len = rank[3], rank[4]
        emb_rank_r_p = (emb.ru_embeddings(rank[5]))
        emb_rank_a_p = (emb.ru_embeddings(rank[6]))
        rank_q_p,rank_q_p_len = rank[7],rank[8]
        

        point=rank[9]



        
        rank_q_output, _ = emb.ubirnn(rank_q, emb.init_hc(rank_q.size(0)))
        rank_q_pad = Variable(torch.zeros(
            rank_q_output.size(0)
            , 1
            , rank_q_output.size(2))).cuda()
        rank_q_output = torch.cat(
            (rank_q_pad, rank_q_output)
            , 1)

        rank_q_len = rank_q_len.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)
        emb_rank_q = rank_q_output.gather(1, rank_q_len.detach())
        
        
        
        
        
        
        
        rank_q_p_output, _ = emb.ubirnn(rank_q_p, emb.init_hc(rank_q_p.size(0)))
        rank_q_p_pad = Variable(torch.zeros(
            rank_q_p_output.size(0)
            , 1
            , rank_q_p_output.size(2))).cuda()
        rank_q_p_output = torch.cat(
            (rank_q_p_pad, rank_q_p_output)
            , 1)

        rank_q_p_len = rank_q_p_len.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)
        emb_rank_q_p = rank_q_p_output.gather(1, rank_q_p_len.detach())
        
        point_rank_mat = torch.stack(
            [emb_rank_r_p, emb_rank_q_p.squeeze(), emb_rank_a_p, ]
            , dim=1) \
            .unsqueeze(1)
        

        point_score = torch.cat([
            self.convnetp1(point_rank_mat)
            , self.convnetp2(point_rank_mat)
            , self.convnetp3(point_rank_mat)]
            , dim=2).squeeze()
        
        


        low_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q.squeeze(), emb_rank_a, ]
            , dim=1) \
            .unsqueeze(1)
        high_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q.squeeze(), emb_rank_acc]
            , dim=1) \
            .unsqueeze(1)

        low_score = torch.cat([
            self.convnet1(low_rank_mat)
            , self.convnet2(low_rank_mat)
            , self.convnet3(low_rank_mat)]
            , dim=2).squeeze()
        high_score = torch.cat([
            self.convnet1(high_rank_mat)
            , self.convnet2(high_rank_mat)
            , self.convnet3(high_rank_mat)]
            , dim=2).squeeze()


        low_score1 = self.fc_new_2(
            self.fc_new_1(low_score.squeeze()).squeeze()).squeeze()
        high_score1 = self.fc_new_2(
            self.fc_new_1(high_score.squeeze()).squeeze()).squeeze()
        
        

        emb_rank_q=emb_rank_q.squeeze(1)
        
        

        
    
        point_score = self.fcp_new_2(
            self.fcp_new_1(point_score.squeeze()).squeeze()).squeeze()


        point_loss = torch.sum(F.sigmoid(abs(point_score - point)))
        rank_loss1 = torch.sum(F.sigmoid(low_score1 - high_score1))

        print("point",point_loss.item())
        print("pairwise",rank_loss1.item())

        print("Rank loss: {:.6f}".format((rank_loss1*0.8+0.2*point_loss ).item()))

        return rank_loss1*0.8+point_loss*0.2

    def test(self, test_data):
        # test_a, _r, _q all variables
        emb = self.embedding_manager
        test_a, test_r, test_q, test_q_len = test_data
        a_size = test_a.size(0)


        emb_rank_a = emb.au_embeddings(test_a)
        emb_rank_r = emb.ru_embeddings(test_r)
        
        test_q_output, _ = emb.ubirnn(test_q.unsqueeze(0), emb.init_hc(1))

        
       
        
        ind = Variable(torch.LongTensor([test_q_len])).cuda()       
        test_q_target_output = torch.index_select(test_q_output.squeeze(), 0, ind)

        
        emb_rank_q = test_q_target_output.squeeze() \
            .repeat(a_size).view(a_size, emb.emb_dim)
        emb_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q, emb_rank_a], dim=1) \
            .unsqueeze(1)

        score = torch.cat([
            self.convnet1(emb_rank_mat)
            , self.convnet2(emb_rank_mat)
            , self.convnet3(emb_rank_mat)]
            , dim=2).squeeze()
        
        point_score = torch.cat([
            self.convnetp1(emb_rank_mat)
            , self.convnetp2(emb_rank_mat)
            , self.convnetp3(emb_rank_mat)]
            , dim=2).squeeze()

        
        
        point_score = self.fcp_new_2(
            self.fcp_new_1(point_score.squeeze()).squeeze()).squeeze()
        
        
        
        score1 = self.fc_new_2(
            self.fc_new_1(score.squeeze()).squeeze()).squeeze()

        
        ret_score=list()
        for i in [0.2]:
            ret_score1 = (point_score*i+score1*(1-i)).data.squeeze().tolist()
            ret_score.append(ret_score1)
        return np.array(ret_score)
