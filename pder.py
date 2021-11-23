

import os
import datetime

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import numpy as np
from embed import Embed
from skipgram import SkipGram
from recsys import RecSys
from data_loader import DataLoader
from utils import Utils
from skipgrame import SkipGrame
from embede import Embede


class PDER:
    def __init__(self, dataset, embedding_dim, epoch_num,
                 batch_size, neg_sample_ratio,
                 lstm_layers, include_content, lr, cnn_channel,
                 test_ratio, lambda_, prec_k,
                 mp_length, mp_coverage, id, answer_sample_ratio):

        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_ratio = neg_sample_ratio
        self.lstm_layers = lstm_layers
        self.learning_rate = lr
        self.test_prop = test_ratio
        self.prec_k = prec_k
        self.cuda1='0'
        
        self.id = id

        self.dl = DataLoader(dataset=dataset
                             , ID=id
                             , include_content=include_content
                             , coverage=mp_coverage
                             , length=mp_length
                             , answer_sample_ratio=answer_sample_ratio
                             )

        self.utils = Utils(dataset=dataset
                           , ID=id
                           , mp_coverage=mp_coverage
                           , mp_length=mp_length
                           )
        if self.dataset=='English':
            print('321412412512512512')
            self.model_folder=os.getcwd()+'/model/'+'eng/'
        else:
            print('fgdasgasdfhgljksahgjklsdhg')
            self.model_folder = os.getcwd() + "/model/"

        print(self.dl.user_count)

        self.embedding_manager = Embed(vocab_size=self.dl.user_count + 1
                                       , embedding_dim=embedding_dim
                                       , lstm_layers=lstm_layers
                                       )
        self.embeddinge_manager = Embede(vocab_size=self.dl.user_count + 1
                                       , embedding_dim=embedding_dim
                                       , lstm_layers=lstm_layers
                                       )

        self.skipgram = SkipGram(embedding_dim=self.embedding_dim
                                 , emb_man=self.embedding_manager
                                 )
#         self.skipgrame = SkipGrame(embedding_dim=self.embedding_dim
#                                  , emb_man=self.embeddinge_manager
#                                  )

        self.recsys = RecSys(embedding_dim=embedding_dim, cnn_channel=cnn_channel
                             , embeddings=self.embedding_manager
                             , embeddingse = self.embeddinge_manager
                             )

    def run(self):
        dl, utils = self.dl, self.utils
        recsys, skipgram = self.recsys, self.skipgram

        if torch.cuda.device_count() > 1:
            print("Using {} GPUs".format(torch.cuda.device_count()))
            skipgram = nn.DataParallel(skipgram)
            skipgrame = nn.DataParallel(skipgrame)
            recsys = nn.DataParallel(recsys)

        if torch.cuda.is_available():  # Check availability of cuda
            print("Using device {}".format(torch.cuda.current_device()))
            skipgram.cuda()
#             skipgrame.cuda()
            recsys.cuda()
        #print("dodoasdasfhdasjkghasdghasdlhk")
        skipgram_optimizer = optim.Adam(skipgram.parameters()
                                        , lr=self.learning_rate)
#         skipgrame_optimizer = optim.Adam(skipgrame.parameters()
#                                         , lr=self.learning_rate)
        recsys_optimizer = optim.Adam(recsys.parameters()
                                      , lr=0.5 * self.learning_rate)

        batch_count = 0
        best_MRR, best_hit_K, best_pa1 = 0, 0, 0
        
        skipgram, skipgram_optimizer, epoch = utils.load_model(model_name="sg", model=skipgram, optimizer=skipgram_optimizer)
        recsys, recsys_optimizer, _ = utils.load_model(model_name="rs", model=recsys, optimizer=recsys_optimizer)

        #for epoch in range(self.epoch_num):
        for epoch in range(epoch, self.epoch_num):
            
            dl.process = True
            iter = 0
            print("epoch num", epoch)
            while dl.process:
                torch.cuda.empty_cache()
#             gg=0
#             while gg<1:
#                 gg=gg+1
                u, v,upos, vpos, npos, aqr, accqr,point_wise, point   = dl.get_train_batch(
                        batch_size=self.batch_size,
                        neg_ratio=self.neg_sample_ratio)
                eupos, evpos, enpos   = dl.get_trainexpert_batch(
                        batch_size=self.batch_size,
                        eu=u,
                        vv=v,
                        neg_ratio=self.neg_sample_ratio)

                if aqr.any():
                    #if aqr[:, 0]:
                    # R-u, R-v, and R-n
                    rupos = torch.tensor(dl.uid2index(upos[0]),device=torch.device('cuda:'+self.cuda1)).long()
                    rvpos = torch.tensor(dl.uid2index(vpos[0]),device=torch.device('cuda:'+self.cuda1)).long()
                    rnpos = torch.tensor(dl.uid2index(npos[0]),device=torch.device('cuda:'+self.cuda1)).long()
                    rpos = [rupos, rvpos, rnpos]
                    
                    reupos = torch.tensor(dl.uid2index(eupos[0]),device=torch.device('cuda:'+self.cuda1)).long()
                    revpos = torch.tensor(dl.uid2index(evpos[0]),device=torch.device('cuda:'+self.cuda1)).long()
                    renpos = torch.tensor(dl.uid2index(enpos[0]),device=torch.device('cuda:'+self.cuda1)).long()
                    repos = [reupos, revpos, renpos]
                    

                    # A-u, A-v, and A-n
                    aupos = torch.tensor(dl.uid2index(upos[1]),device=torch.device('cuda:'+self.cuda1)).long()
                    avpos = torch.tensor(dl.uid2index(vpos[1]),device=torch.device('cuda:'+self.cuda1)).long()
                    anpos = torch.tensor(dl.uid2index(npos[1]),device=torch.device('cuda:'+self.cuda1)).long()
                    apos = [aupos, avpos, anpos]
                    
                    aeupos = torch.tensor(dl.uid2index(eupos[1]),device=torch.device('cuda:'+self.cuda1)).long()
                    aevpos = torch.tensor(dl.uid2index(evpos[1]),device=torch.device('cuda:'+self.cuda1)).long()
                    aenpos = torch.tensor(dl.uid2index(enpos[1]),device=torch.device('cuda:'+self.cuda1)).long()
                    aepos = [aeupos, aevpos, aenpos]
                    

                    qu_wc = dl.qid2padded_vec(upos[2])
                    qv_wc = dl.qid2padded_vec(vpos[2])
                    qn_wc = dl.qid2padded_vec(npos[2])

                    qulen = dl.qid2vec_length(upos[2])
                    qvlen = dl.qid2vec_length(vpos[2])
                    #print("1",len(npos[2]))
                    qnlen = dl.qid2vec_length(npos[2])
                    

                    
                    
                    
                    qu_wc = torch.tensor(qu_wc,device=torch.device('cuda:'+self.cuda1)).float().view(-1, dl.PAD_LEN, 300)
                    qv_wc = torch.tensor(qv_wc,device=torch.device('cuda:'+self.cuda1)).float().view(-1, dl.PAD_LEN, 300)
                    qn_wc = torch.tensor(qn_wc,device=torch.device('cuda:'+self.cuda1)).float().view(-1, dl.PAD_LEN, 300)
                    

                    
                    qulen = torch.tensor(qulen,device=torch.device('cuda:'+self.cuda1)).long()
                    qvlen = torch.tensor(qvlen,device=torch.device('cuda:'+self.cuda1)).long()
                    qnlen = torch.tensor(qnlen,device=torch.device('cuda:'+self.cuda1)).long()
                    qinfo = [qu_wc, qv_wc, qn_wc, qulen, qvlen, qnlen]
                    
                    
                    
                    equ_wc = qu_wc#dl.qid2padded_vec(eupos[2])
                    eqv_wc = dl.qid2padded_vec(evpos[2])
                    eqn_wc = dl.qid2padded_vec(enpos[2])

                    equlen = qulen#dl.qid2vec_length(eupos[2])
                    eqvlen = dl.qid2vec_length(evpos[2])
                    #print("2",len(enpos[2]))                       
                    eqnlen = dl.qid2vec_length(enpos[2])

                    equ_wc = qu_wc#Variable(torch.FloatTensor(equ_wc).view(-1, dl.PAD_LEN, 300))
                    eqv_wc = torch.tensor(eqv_wc,device=torch.device('cuda:'+self.cuda1)).float().view(-1, dl.PAD_LEN, 300)
                    eqn_wc = torch.tensor(eqn_wc,device=torch.device('cuda:'+self.cuda1)).float().view(-1, dl.PAD_LEN, 300)
                    equlen = qulen#Variable(torch.LongTensor(equlen))
                    eqvlen = torch.tensor(eqvlen,device=torch.device('cuda:'+self.cuda1)).long()
                    eqnlen = torch.tensor(eqnlen,device=torch.device('cuda:'+self.cuda1)).long()
                    eqinfo = [equ_wc, eqv_wc, eqn_wc, equlen, eqvlen, eqnlen]

                    # aqr: R, A, Q

                    #print("aqr",aqr)
                    rank_r = torch.tensor(dl.uid2index(aqr[:, 0]),device=torch.device('cuda:'+self.cuda1)).long()
                    rank_a = torch.tensor(dl.uid2index(aqr[:, 1]),device=torch.device('cuda:'+self.cuda1)).long()
                    rank_acc = torch.tensor(dl.uid2index(accqr),device=torch.device('cuda:'+self.cuda1)).long()
                    
                    rank_q_wc = dl.qid2padded_vec(aqr[:, 2])
                    rank_q_len = dl.qid2vec_length(aqr[:, 2])
                    rank_q_time = dl.qid2vec_t(aqr[:, 2])
                    rank_q = torch.tensor(rank_q_wc,device=torch.device('cuda:'+self.cuda1)).float().view(-1, dl.PAD_LEN, 300)
                    rank_q_len = torch.tensor(rank_q_len,device=torch.device('cuda:'+self.cuda1)).long()
                    rank_q_t =torch.tensor(rank_q_time,device=torch.device('cuda:'+self.cuda1)).float().view(-1, 256)
                    
                    
                    rank_r_p=torch.tensor(dl.uid2index(point_wise[:, 0]),device=torch.device('cuda:'+self.cuda1)).long()
                    rank_a_p=torch.tensor(dl.uid2index(point_wise[:, 1]),device=torch.device('cuda:'+self.cuda1)).long()
                    rank_q_p_wc = dl.qid2padded_vec(point_wise[:, 2])
                    rank_q_p_len = dl.qid2vec_length(point_wise[:, 2])
                    rank_q_p = torch.tensor(rank_q_p_wc,device=torch.device('cuda:'+self.cuda1)).float().view(-1, dl.PAD_LEN, 300)
                    rank_q_p_len = torch.tensor(rank_q_p_len,device=torch.device('cuda:'+self.cuda1)).long()
                    
                    
                    
                    point=torch.tensor(point,device=torch.device('cuda:'+self.cuda1)).long()
                    

                    rank = [rank_r, rank_a, rank_acc, rank_q, rank_q_len,rank_r_p ,rank_a_p,rank_q_p,rank_q_p_len,point]

#                     if torch.cuda.is_available():
#                         rpos = [x.cuda() for x in rpos]
#                         apos = [x.cuda() for x in apos]
#                         qinfo = [x.cuda() for x in qinfo]
#                         repos = [x.cuda() for x in repos]
#                         aepos = [x.cuda() for x in aepos]
#                         eqinfo = [x.cuda() for x in eqinfo]
                        
                        
#                         rank = [x.cuda() for x in rank]


                    #print("rupos",rpos[0].size())

                    cur_time = str(datetime.datetime.now())
                    print("{:s}, E:{:d}, I{:d}".format(cur_time, epoch, iter), end=" ")

                    """
                    ============== Skip-gram ===============
                    """
                    skipgram_optimizer.zero_grad()
                    skipgram.train()
                    skipgram_loss = skipgram(rpos=rpos, apos=apos
                                             , qinfo=qinfo,
                                             repos=repos
                                             , aepos=aepos
                                             , eqinfo=eqinfo)
                    skipgram_loss.backward()
                    skipgram_optimizer.step()
                    
                    ##print("1")
                    
#                     skipgrame_optimizer.zero_grad()
#                     skipgrame.train()
#                     skipgrame_loss = skipgrame(rpos=repos
#                                              , apos=aepos
#                                              , qinfo=eqinfo)
#                     skipgrame_loss.backward()
#                     skipgrame_optimizer.step()


    #   need              """
    #                 ============== Rec-Sys ===============
    #                 """
                    recsys_optimizer.zero_grad()
                    recsys.train()
                    recsys_loss = recsys(rank=rank)
                    recsys_loss.backward()
                    recsys_optimizer.step()

                    iter += 1
                    batch_count += 1
                    #print('tesrsdtasdtsd')
                    """
                    Ideas of printing validation, recording performance, 
                        and dumping model.
                        1 - Ea. 5: print epoch, iter, time 
                        2 - Ea. 20: print iter, MRR, haK, pa1, sampled vald set
                        3 - Ea. 100: record iMRR, haK, pa1, sampled vald set
                        4 - Ea. 1000: check if better result, dump model, all valid set
                        5 - ea. epoch: print, record
                    """
                    n_sample = upos.shape[1]

                    # Print training progress every 10 iterations
                    #if iter % 10 == 0:
                    #    tr = datetime.datetime.now().isoformat()[8:24]
                    #    print("E:{}, I:{}, size:{}, {}, Loss:{:.3f}"
                    #          .format(epoch, iter, n_sample, tr, skipgram_loss.data[0]))

                    # Write to file every 10 iterations
                    if batch_count % 200== 0:
                        # hMRR, hhit_K, hpa1 = 0, 0, 0
                        before_train_time = datetime.datetime.now()
                        hMRR, hhit_K, hpa1 = self.test()
                        #print("12345",hMRR,hhit_K,hpa1)
                        cur_time_dt = datetime.datetime.now()
                        delta_time = (cur_time_dt - before_train_time).total_seconds()
                        cur_time = str(cur_time_dt)
                        print("\t",cur_time," Entire Val@ I:",iter, 'MRR=',hMRR," hitK=",hhit_K,' pa1=',hpa1)
                        msg = (cur_time, delta_time, batch_count, epoch, iter, hMRR, hhit_K, hpa1, 
                                      skipgram_loss.item(), recsys_loss.item())
                        utils.write_performance(msg=msg)
                    if batch_count % 200 == 0:
                        # hMRR, hhit_K, hpa1 = 0, 0, 0
                        before_train_time = datetime.datetime.now()
                        hMRR, hhit_K, hpa1 = self.traintest()
                        cur_time_dt = datetime.datetime.now()
                        delta_time = (cur_time_dt - before_train_time).total_seconds()
                        cur_time = str(cur_time_dt)
                        print("\t traintest ",cur_time," Entire Val@ I:",iter, 'MRR=',hMRR," hitK=",hhit_K,' pa1=',hpa1)

                        msg = ("traintest," , cur_time, delta_time, batch_count, epoch, iter, hMRR, hhit_K, hpa1, 
                                      skipgram_loss.item(), recsys_loss.item())
                        utils.write_performance(msg=msg)
                        
                        

#                     if batch_count % 1000 == 0:
#                         with open("./performance/ptt", "w") as fout:
#                             fout.write("\n\n")
#                             fout.write(" ".join([str(x) for x in all_scores]))

                    if batch_count % 200 == 0:
                        print('strat saving model!!!')
                        utils.save_model(model_name="sg", model=skipgram, optimizer=skipgram_optimizer, epoch=epoch)
                        utils.save_model(model_name="rs", model=recsys, optimizer=recsys_optimizer, epoch=epoch)
                        print('saving done!')

            eMRR, ehit_K, epa1 = self.test()
            print("Entire Val@ E:",epoch, "MRR-",eMRR," hit_K-",ehit_K, "pa1-",epa1)
            msg = (epoch, iter, eMRR, ehit_K, epa1)
            utils.write_performance(msg=msg)

        print("Optimization Finished!")

    def test(self, test_prop=None):
        model, dl = self.recsys, self.dl
        model.eval()
        #MRR, hit_K, prec_1 = 0, 0, 0

        test_batch = dl.get_test_batch(test_prop=test_prop)
        test_batch_len = len(test_batch)

        #all_scores = []

        # The format of test_batch is:  [aids], rid, qid, accid
        for rid, qid, accaid, aid_list in test_batch:
            rank_a = Variable(torch.LongTensor(dl.uid2index(aid_list)))
            rep_rid = [rid] * len(aid_list)
            rank_r = Variable(torch.LongTensor(dl.uid2index(rep_rid)))
            rank_q_len = dl.q2len(qid)
            #print(qid.shape)
            rank_q_t=Variable(torch.FloatTensor(dl.q2time(qid)))
            rank_q = Variable(torch.FloatTensor(dl.q2emb(qid)))

            #print("dsafasd",accaid)
            #rank_a_s=dl.aids2sids(np.array(aid_list))
            #rank_acc_s=dl.aid_get_sids(accaid)

            
            if torch.cuda.is_available():
                rank_a = rank_a.cuda()
                rank_r = rank_r.cuda()
                rank_q = rank_q.cuda()
                rank_q_t=rank_q_t.cuda()


                

            model.eval()
            score = model.test(test_data=[rank_a, rank_r, rank_q, rank_q_len,rank_q_t])

#             try:
#                 all_scores = all_scores+score
#             except NameError:
#                 all_scores = score
            MRR_list=list()
            hit_K_list=list()
            prec_1_list=list()
            #all_scores = all_scores+score
            for i in score:
                try:#print(aid_list)
                    RR, hit, prec = self.utils.performance_metrics(aid_list
                                                                   , i
                                                                   , accaid
                                                                   , self.prec_k)
                except TypeError: 
                    print(aid_list,score,accaid,self.prec_k)
               
                               
                               

                MRR_list.append(RR)
                
                hit_K_list.append(hit)
                prec_1_list.append(prec)

            MRR_list=np.array(MRR_list)
            hit_K_list=(np.array(hit_K_list)).astype(int)
            prec_1_list=np.array(prec_1_list)
            try:
                MRR = MRR+MRR_list
                hit_K=hit_K+hit_K_list
                prec_1=prec_1+prec_1_list
            except NameError:
                #print("not found")
                MRR= MRR_list
                hit_K=hit_K_list
                prec_1=prec_1_list

              

        MRR, hit_K, prec_1 = MRR / test_batch_len, hit_K / test_batch_len, prec_1 / test_batch_len
        return MRR, hit_K, prec_1#, all_scores

    def traintest(self, test_prop=None):
        model, dl = self.recsys, self.dl
        model.eval()
        #MRR, hit_K, prec_1 = 0, 0, 0

        test_batch = dl.get_traintest_batch(test_prop=test_prop)
        test_batch_len = len(test_batch)

        all_scores = []

        # The format of test_batch is:  [aids], rid, qid, accid
        for rid, qid, accaid, aid_list in test_batch:
            rank_a = Variable(torch.LongTensor(dl.uid2index(aid_list)))
            rep_rid = [rid] * len(aid_list)
            rank_r = Variable(torch.LongTensor(dl.uid2index(rep_rid)))
            rank_q_len = dl.q2len(qid)
            rank_q = Variable(torch.FloatTensor(dl.q2emb(qid)))
            rank_q_t=Variable(torch.FloatTensor(dl.q2time(qid)))
            #print("dsafasd",accaid)
            #rank_a_s=dl.aids2sids(np.array(aid_list))
            #rank_acc_s=dl.aid_get_sids(accaid)

            
            if torch.cuda.is_available():
                rank_a = rank_a.cuda()
                rank_r = rank_r.cuda()
                rank_q = rank_q.cuda()
                rank_q_t=rank_q_t.cuda()


                

            model.eval()
            score = model.test(test_data=[rank_a, rank_r, rank_q, rank_q_len,rank_q_t])
#             try:
#                 all_scores = all_scores+score
#             except NameError:
#                 all_scores = score
            MRR_list=list()
            hit_K_list=list()
            prec_1_list=list()
            #all_scores = all_scores+score
            for i in score:
                try:#print(aid_list)
                    RR, hit, prec = self.utils.performance_metrics(aid_list
                                                                   , i
                                                                   , accaid
                                                                   , self.prec_k)
                except TypeError: 
                    print(aid_list,score,accaid,self.prec_k)
               
                MRR_list.append(RR)
                hit_K_list.append(hit)
                prec_1_list.append(prec)
            MRR_list=np.array(MRR_list)
            hit_K_list=(np.array(hit_K_list)).astype(int)
            prec_1_list=np.array(prec_1_list)
            try:
                MRR = MRR+MRR_list
                hit_K=hit_K+hit_K_list
                prec_1=prec_1+prec_1_list
            except NameError:
                MRR= MRR_list
                hit_K=hit_K_list
                prec_1=prec_1_list

              

        MRR, hit_K, prec_1 = MRR / test_batch_len, hit_K / test_batch_len, prec_1 / test_batch_len
        return MRR, hit_K, prec_1#, all_scores
