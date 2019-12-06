import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class HiLSTM(nn.Module):
    def __init__(self, args, num_labels=6, device='cpu'):
        super(HiLSTM, self).__init__()
        self.embed_size = args.embed_size
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size

        self.device = device

        self.embed = nn.Embedding(args.vocab_size, args.embed_size)

        self.lstmP = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.hidden_size)
        self.lstmS = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,batch_first=True)
        
        #第一个Linear V_s -> E_s^V
        self.MLP = nn.Linear(num_labels, self.hidden_size)
        #第二个Linear V_s -> E_s^V
        self.MLP2 = nn.Linear(2*num_labels, self.hidden_size)
        self.Classifier = nn.Linear(self.hidden_size, self.vocab_size)

        self.attn = nn.Linear(2*self.hidden_size, self.hidden_size)

    def forward(self, V_s, ViList, input_id, max_aspect=10):
        hp_0 = self.MLP(V_s)   # LSTMP 的初始状态
        xp_0 = torch.zeros(self.batch_size, self.embed_size).to(self.device) 
        cp_0 = torch.zeros(self.batch_size, self.hidden_size).to(self.device)
        overall_logits = None

        for i in range(max_aspect):
            hp_n, cp_n = self.lstmP(xp_0, (hp_0, cp_0))
            V_i = ViList[:,i,:]             # Vi 是 Vs concat一个label的one-hot

            E_i = self.MLP2(V_i)            
        
            hs_0 = self.attn(torch.cat((hp_n, E_i), 1)) 
            cs_0 = torch.zeros(self.batch_size, self.hidden_size).to(self.device)
            
            word = self.embed(input_id[:, i, :])
            
            output, (hs_n, _) = self.lstmS(word, (hs_0.unsqueeze(0), cs_0.unsqueeze(0)))
            
            logits = self.Classifier(output) 
            if i == 0:
                overall_logits = logits
            else:
                overall_logits = torch.cat((overall_logits, logits), 1)
            
            xp_0 = hs_n.squeeze(0)
            hp_0 = hp_n
            cp_0 = cp_n
        return overall_logits

    def generate(self, V_s, ViList, max_aspect=10, beginning_index=None):
        self.batch_size = 1
        hp_0 = self.MLP(V_s)
        xp_0 = torch.zeros(self.batch_size, self.embed_size).to(self.device)
        cp_0 = torch.zeros(self.batch_size, self.hidden_size).to(self.device)

        all_index = []

        for i in range(max_aspect):           
            hp_n, cp_n = self.lstmP(xp_0, (hp_0, cp_0))
            V_i = ViList[:, i, :]
            E_i = self.MLP2(V_i)

            hs_0 = self.attn(torch.cat((hp_n, E_i),1))
            cs_0 = torch.zeros(self.batch_size, self.hidden_size).to(self.device)
            history = torch.LongTensor([beginning_index]).unsqueeze(0) 
            print(history)
            for j in range(100):
                word = self.embed(history)
                output, (hs_n, _) = self.lstmS(word, (hs_0.unsqueeze(0), cs_0.unsqueeze(0)))
                print(output.shape)
                logits = self.Classifier(output[:,-1,:])
                
                logits = F.softmax(logits[0])
 
                index = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                print("logits",logits)
                print("index",index)
                print("history",history)
                history = torch.cat((history, index),1)
                print("index",index.data)
                print("history",history)
                
            all_index.append(history)
            print("index",all_index)








            





















