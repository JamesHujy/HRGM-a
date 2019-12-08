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

def train(model, args, train_iter, valid_iter, ignore_index=None):
	print(model)
	
	optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
	criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
	best_valid_loss = 1e15
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

			logits = model(Vs, Vi, sentence, batch_size=args.batch_size)
			cat_sentence = None
			for i in range(args.num_aspect):
				if i == 0:
					cat_sentence = sentence[:,i,:]
				else:
					cat_sentence = torch.cat((cat_sentence, sentence[:,i,:]), dim=1)
			loss = None

			for i in range(219):
				if i == 0:
					loss = criterion(logits[:,i,:], cat_sentence[:, i+1])
				else:
					loss += criterion(logits[:,i,:], cat_sentence[:, i+1])

			loss.backward()
			
			optimizer.step()
			steps += 1
			if steps % args.logger_interval == 0:
				print('epoch{}: steps[{}] - loss: {:.6f}'.format(epoch, steps, loss.item()))
		#test(model, args, train_iter, epoch)
		with torch.no_grad():
			valid_loss = valid(model, valid_iter, args,criterion)
			if valid_loss < best_valid_loss:
				save(model, args, epoch="final")
				best_loss = valid_loss

def valid(model, valid_iter,args,criterion):
	total_loss = 0
	for i, (Vs, Vi, sentence) in enumerate(valid_iter):
		model.eval()
		sentence = Variable(sentence.long())
		Vs = Variable(Vs.float())
		Vi = Variable(Vi.float())
		if torch.cuda.is_available():
			sentence, Vs, Vi = sentence.cuda(), Vs.cuda(), Vi.cuda()
		
		logits = model(Vs, Vi, sentence, batch_size=1)
		
		cat_sentence = None
		for i in range(args.num_aspect):
			if i == 0:
				cat_sentence = sentence[:,i,:]
			else:
				cat_sentence = torch.cat((cat_sentence, sentence[:,i,:]), dim=1)
		loss = None
		for i in range(219):
			if i == 0:
				loss = criterion(logits[:,i,:], cat_sentence[:, i+1])
			else:
				loss += criterion(logits[:,i,:], cat_sentence[:, i+1])
		total_loss += loss
	print()
	print("----------valid loss: {} ---------".format(total_loss/len(valid_iter)))
	print()
	return total_loss

def test(model, args, test_iter, path, index2char, beginning_index=None):
	print("loading model")
	model.load_state_dict(torch.load(path))
	reviews = []
	dis = []
	for (Vs, Vi, sentence) in test_iter:
		all_index = model.generate(Vs, Vi ,beginning_index=beginning_index)
		all_content = ""
		for content in all_index:
			token = ' '.join(index2char(content[1:]))
			all_content+=token
			all_content+=" "
			dis.append(token)
		print(all_content)
		reviews.append(all_content)
	with open(args.dataset+"_review.txt",'w') as f:
		f.write('\n'.join(reviews))
	with open(args.dataset+"_dis.txt",'w') as f:
		f.write('\n'.join(dis))

def save(model, args, epoch):
	if not os.path.isdir(args.save_dir):
		os.makedirs(args.save_dir)
	save_path = args.save_dir + "model_"+str(epoch)+".pt"
	torch.save(model.state_dict(), save_path)




