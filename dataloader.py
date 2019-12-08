import pickle
import json
from torch.utils.data import Dataset
import numpy as np
import torch

class DataProcess(Dataset):
    def __init__(self, label_num=6, batch_size=1, dataset=None,filename="ltrain_1121.json", fileprefix="/data/disk3/private/chenhuimin/rewgen/code/rewgen_dual_pral/data/train_rb/"):
        self.prefix = fileprefix
        self.filename = filename
        self.textList = []
        self.aslabelsList = []
        self.ratelabelList = []
        self.VsList = []
        self.dataset = dataset
        self.stars = []
        self.ViList = []
        self.label_num = label_num
        self.vocab = pickle.load(open(self.prefix + "vocab.pickle",'rb'))
        self.idx2char = {value:keys for (keys, value) in self.vocab.items()}
        self.tokenized = []
        self.batch_size = batch_size
        self.read_data()
        self.get_Vs()
        self.get_Vi()
        self.char2idx()
        

    def get_padding_index(self):
        return self.vocab['PAD']

    def get_beginning_index(self):
        return self.vocab['SGO']

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
            if self.dataset is not None:
                self.write_data()
            if self.dataset is not None:
                self.write_aspect()

            if 'asplabels' in rewdic and 'ratelabels' in rewdic:
                aslabels = rewdic['asplabels']
                ratelabels = rewdic['ratelabels']
                self.aslabelsList.append(aslabels)
                self.ratelabelList.append(ratelabels)
            
    def write_aspect(self):
        with open(self.prefix + self.filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            rewdic = json.loads(line)
            self.stars.append(rewdic['stars'])
        
        index = 0
        contentList = []
        with open(self.dataset+"_dis.txt",'r') as f:
            lines = f.readlines()

        for (aslabels, ratelabels, stars) in zip(self.aslabelsList, self.ratelabelList, self.stars):
            for (aspect, rate) in zip(aslabels, ratelabels):
                content = "00#00#{}#{}#{}#{}####".format(stars, aspect, rate, lines[index].strip())
                contentList.append(content)
                index += 1 
        with open("testfile_ctrl_{}.txt".format(self.dataset), 'w') as f:
            f.write('\n'.join(contentList))

    def write_data(self):
        review = []
        for sen in self.textList:
            all_content = ""
            for part in sen:
                all_content += " ".join(part)
                all_content += " "
            review.append(all_content)
        with open(self.dataset+"_ref.txt",'w') as f:
            f.write('\n'.join(review))
        
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
                idxes.append(self.vocab['</S>'])
                partly.append(idxes)
            tokenized.append(partly)
        self.tokenized = tokenized
        self.paddingsentence()

    def id2char(self, index):
        ans = []
        for idx in index:
            if idx == self.vocab['</S>']:
                return ans
            ans.append(self.idx2char[idx])
        return ans

    def paddingsentence(self, maxaspect=10, maxlength=22):
        #一句话padding到长度为20
        padding_text = []
        for sentence in self.tokenized:
            partly = []
            for part in sentence:
                zeros = [self.vocab['PAD']] * (maxlength - len(part))
                part.extend(zeros)
                partly.append(part)

            for i in range(maxaspect - len(sentence)):
                padding = [self.vocab['PAD']]*(maxlength)
                partly.append(padding)  
            
            padding_text.append(partly)
        self.tokenized = padding_text
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


        









