import os, sys
import torch
import numpy as np
class Utils:
    def __init__(self, dataset, ID, mp_length, mp_coverage):
        self.dataset = dataset
        self.id = ID
        if self.dataset=='English':
            self.model_folder=os.getcwd()+'/model/'+'eng/'
        else:
            self.model_folder = os.getcwd() + "/model/"

        self.PERF_DIR = os.getcwd() + "/performance/"
        self.performance_file = self.PERF_DIR + \
                                "{}_{}_{}_{}.txt".format(self.dataset, str(self.id),
                                                         str(mp_length), str(mp_coverage))
        pass

    def performance_metrics(self, aid_list, score_list, accid, k_s):
        """
        Performance metric evaluation

        Args:
            aid_list  -  the list of aid in this batch
            score_list  -  the list of score of ranking
            accid  -  the ground truth
            k  -  precision at K
        """
        k_s=[5]
        if len(aid_list) != len(score_list):
            print("aid_list and score_list not equal length.",
                  file=sys.stderr)
            sys.exit()
        id_score_pair = list(zip(aid_list, score_list))

        id_score_pair.sort(key=lambda x: x[1], reverse=True)
        ss=0

        for ind, (aid, score) in enumerate(id_score_pair):

            if aid == accid:
                ss=1
                if ind == 0:
                    return 1/(ind+1), (ind < k_s[0]), 1
                    #return 1/(ind+1), np.array([int(ind < k) for k in k_s]), 1
                else:
                    return 1/(ind+1), (ind < k_s[0]), 0
                    #return 1/(ind+1), np.array([int(ind < k) for k in k_s]), 0
        if ss==0:
            return 0,0,0
            #return 0,np.array([int(100 < k) for k in k_s]),0
                

    def save_model(self, model_name, model, optimizer, epoch):
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, "{}{}_{}.checkpoint".format(self.model_folder, model_name, str(self.id)))
        return
    
    def load_model(self, model_name, model, optimizer):
        model_file = "{}{}_{}.checkpoint".format(self.model_folder, model_name, str(self.id))
        epoch = 0
        if os.path.exists(model_file):
            print(f'loading checkpoing from {model_file}')
            checkpoint = torch.load(model_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
        
        return model, optimizer, epoch


    def write_performance(self, msg):
        if not os.path.exists(self.PERF_DIR):
            os.mkdir(self.PERF_DIR)
            with open(self.performance_file,"w") as fout:
                print("Epoch,Iter,MRR,hit_K,pa1", file=fout)
        with open(self.performance_file, "a") as fout:
            print(msg, file=fout)
