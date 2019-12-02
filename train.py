import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import tqdm
from sklearn.metrics import f1_score

def train(model, args, train_iter, valid_iter):
	print(model)
	
	optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
	criterion = nn.CrossEntropyLoss(weight=None)
	
	for epoch in range(1, args.epochs+1):
		epochloss = 0
		steps = 0

		for Vs, Vi, sentence in train_iter:
			model.train()
			optimizer.zero_grad()
			sentence = Variable(sentence.long())
			Vs = Variable(Vs.float())
			Vi = Variable(Vi.float())

			if torch.cuda.is_available():
				sentence, Vs, Vi = sentence.cuda(), Vs.cuda(), Vi.cuda()

			logits = model(Vs, Vi, sentence)
			cat_sentence = None
			for i in range(args.num_aspect):
				if i == 0:
					cat_sentence = sentence[:,i,:]
				else:
					cat_sentence = torch.cat((cat_sentence, sentence[:,i,:]), dim=1)
			loss = None
			for i in range(210):
				if i == 0:
					loss = criterion(logits[:,i,:], cat_sentence[:,i])
				else:
					loss += criterion(logits[:,i,:], cat_sentence[:,i])
			
			loss.backward()
			
			optimizer.step()
			steps += 1
			if steps % args.logger_interval == 0:
				print('epoch{}: steps[{}] - loss: {:.6f}'.format(epoch, steps, loss.item()))
			
		if epoch > 15:
			save(model, args.save_dir, epoch)

def test(model, args, test_iter, path):
	print("loading model")
	model.load_state_dict(torch.load(path))
	for (Vs, Vi, sentence) in test_iter:
		model.generate(Vs, Vi)


def save(model, save_dir, epoch):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_path = '{}epoch_{}.pt'.format(save_dir, epoch)
	torch.save(model.state_dict(), save_path)




