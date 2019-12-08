import train
from dataloader import DataProcess
from model import HiLSTM
import argparse
from torch.utils.data import Dataset, DataLoader
import os
import torch
def main():
	parser = argparse.ArgumentParser(description='HRGM')
	parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
	parser.add_argument('-epochs', type=int, default=1000, help='number of epochs for train [default: 100]')
	parser.add_argument('-batch_size', type=int, default=10, help='batch size for training [default: 10]')
	parser.add_argument('-logger-interval',  type=int, default=10,   help='how many steps to wait to log training status')
	parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
	parser.add_argument('-save-interval', type=int, default=50, help='how many steps to wait before saving [default:500]')
	parser.add_argument('-save_dir', type=str, default='./checkpoints/', help='where to save the snapshot')
	parser.add_argument('-max_length', type=int, default=20, help='whether to save when get best performance')
	parser.add_argument('-model_path', type=str, default='./checkpoints/cnn_300_0.001_epoch20.pt', help='set the model saved path to load model')
	# data
	parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
	# model
	parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
	parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
	parser.add_argument('-embed-size', type=int, default=512, help='number of embedding dimension [default: 300]')
	parser.add_argument('-hidden_size', type=int, default=512, help='set the LSTM hidden_size')
	parser.add_argument('-vocab_size', type=int, default=30716, help='set the LSTM hidden_size')
	parser.add_argument('-num_aspect', type=int, default=10, help='set the LSTM hidden_size')
	parser.add_argument('-n_layers', type=int, default=2, help='set the number of layers')
	parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
	# device
	parser.add_argument('-device', type=str, default=2, help='device to use for iterate data, -1 mean cpu [default: -1]')
	# option
	parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
	parser.add_argument('-test', type=bool, default=False, help='train or test')
	parser.add_argument('-dataset', type=str, default="yelp", help='choose dataset to train the model')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.device
	
	dataprefix = None
	filename = None

	args.save_dir = "./checkpoints_"+args.dataset+"/"

	if args.dataset == "yelp":
		dataprefix = "/data/disk3/private/chenhuimin/rewgen/data/yelp/data_1129/"
		filename = {'train':"ltrain_1129.json", "valid":"lvalid_1129.json", "test":"ltest_1129.json"}
	else:
		filename = {'train':"ltrain_1121.json", "valid":"lvalid_1121.json", "test":"ltest_1121.json"}
		dataprefix = "/data/disk3/private/chenhuimin/rewgen/code/rewgen_dual_pral/data/train_rb/"

	training_set = DataProcess(batch_size=args.batch_size, filename=filename['train'], fileprefix=dataprefix)
	training_iter = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=True)

	valid_set = DataProcess(filename=filename['valid'], fileprefix=dataprefix)
	valid_iter = DataLoader(dataset=valid_set, batch_size=1, shuffle=True)

	test_set = DataProcess(dataset=args.dataset, filename=filename['test'],fileprefix=dataprefix)
	test_iter = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

	ignore = training_set.get_padding_index()
	beginning = training_set.get_beginning_index()

	if args.test:
		device = "cpu"
		print("testing model")
		model = HiLSTM(args, device=device)
		model.to(device)
		train.test(model, args, test_iter, index2char=test_set.id2char, path="./checkpoints_"+args.dataset+"/model_final.pt", beginning_index=beginning)
	else:
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = HiLSTM(args, device=device)
		model.to(device)
		train.train(model, args, training_iter, valid_iter, ignore_index=ignore)

if __name__ == '__main__':
	main()




