import pickle
import json
from torch.utils.data import Dataset
import numpy as np
import torch

class DataProcess(Dataset):
    def __init__(self, label_num=6, batch_size=1, filename="ltrain_1121.json", fileprefix="/data/disk3/private/chenhuimin/rewgen/code/rewgen_dual_pral/data/train_rb/"):
        self.prefix = fileprefix
        self.filename = filename
        self.textList = []
        self.aslabelsList = []
        self.ratelabelList = []
        self.VsList = []
        self.ViList = []
        self.label_num = label_num
        self.vocab = pickle.load(open(self.prefix + "vocab.pickle",'rb'))
        self.idx2char = { value:keys for (keys, value) in self.vocab.items()}
        self.tokenized = []
        self.batch_size = batch_size
        try:
            self.load()
        except:
            self.read_data()
            self.get_Vs()
            self.get_Vi()
            self.char2idx()
    
    def load(self):
        self.VsList = pickle.load(open("VsList.pickle",'r'))
        self.ViList = pickle.load(open("ViList.pickle",'r'))
        self.tokenized = pickle.load(open("tokenized.pickle",'r'))
        print("loading finish")

    def read_data(self):
        with open(self.prefix + self.filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            rewdic = json.loads(line)
            if 'words' in rewdic:
                text = rewdic['words']
            else:
                text = rewdic['text']
            self.textList.append(text)

            if 'asplabels' in rewdic and 'ratelabels' in rewdic:
                aslabels = rewdic['asplabels']
                ratelabels = rewdic['ratelabels']
                self.aslabelsList.append(aslabels)
                self.ratelabelList.append(ratelabels)
        
    def get_Vs(self):
        totalList = []
        for (aslabel, ratelabel) in zip(self.aslabelsList, self.ratelabelList):
            initial_vector = [0]*self.label_num
            labelcount = [0] * self.label_num
            for (label, rate) in zip(aslabel, ratelabel):
                if(labelcount[label] > 0):
                    initial_vector[label] = ((initial_vector[label] * labelcount[label]) + (rate))/(labelcount[label] + 1)
                    labelcount[label] += 1
                else:
                    initial_vector[label] = (rate + 1)
                    labelcount[label] += 1
            totalList.append(initial_vector)
        self.VsList = totalList
        with open("VsList.pkl", "wb") as f:
            pickle.dump(self.VsList, f)

    def get_Vi(self):
        ViList = []
        for (Vs, aslabel) in zip(self.VsList, self.aslabelsList):
            tempList = []
            for label in aslabel:
                temp = [0] * self.label_num
                temp[label] = 1
                temp[0:0] = Vs
                tempList.append(temp)
            ViList.append(tempList)
        self.ViList = ViList
        self.paddingaspect()

    def char2idx(self):
        tokenized = []
        maxlength = 0
        for sen in self.textList:
            partly = []
            for partsen in sen:
                idxes = []
                idxes.append(self.vocab['SGO'])
                for char in partsen:
                    try:
                        idx = self.vocab[char]
                    except:
                        idx = self.vocab['UNK']
                    idxes.append(idx)
                partly.append(idxes)
            tokenized.append(partly)
        self.tokenized = tokenized
        self.paddingsentence()

    def idx2char(self, index):
        ans = []
        for idx in index:
            ans.append(self.idx2char[idx])

    def paddingsentence(self, maxaspect=10, maxlength=21):
        padding_text = []
        for sentence in self.tokenized:
            partly = []
            for part in sentence:
                zeros = [30715] * (maxlength - len(part))
                part.extend(zeros)
                part.append(self.vocab['</S>'])
                partly.append(part)

            
            for i in range(maxaspect - len(sentence)):
                
                padding = [30715]*(maxlength + 1)
                
                partly.append(padding)  
                         
            padding_text.append(partly)
        self.tokenized = padding_text
        #print(np.array(self.tokenized).shape)
        with open("tokenized.pkl", "wb") as f:
            pickle.dump(self.tokenized, f)

    def paddingaspect(self, maxaspect=10):
        aspect = []
        for aslabel in self.ViList:
            length = len(aslabel)
            for i in range(maxaspect - length):
                aslabel.append([-1]*2*self.label_num)
            aspect.append(aslabel)
        self.ViList = aspect
        with open("ViList.pkl", "wb") as f:
            pickle.dump(self.ViList, f)
        
    def __len__(self):
        return self.batch_size*(len(self.tokenized)//self.batch_size)

    def __getitem__(self, item):
        return torch.Tensor(self.VsList[item]), torch.Tensor(self.ViList[item]), torch.Tensor(self.tokenized[item])

def main():
    dataloader = DataProcess()

if __name__ == "__main__":
    main()


        









