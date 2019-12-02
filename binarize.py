#coding=utf-8
#通过更改mask和label来控制是否是有监督数据
import pickle
import argparse
import random
import json


parser = argparse.ArgumentParser(
    description="""
preprocess the corpus,
creating vocab, ivocab,
creating binary text,
the intput file should as follows:
each line is a poem sentence, and each sentence should be splited by white space, based on Chinese character.
four continuous sentences belong to a same quatrain
""",  formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument("-b", "--binarized", default='dev_corpus.pkl',
                    help="the name of the pickled binarized text file")
parser.add_argument("-i", "--input", default="corpus.json",
                    help="the input file")
parser.add_argument("-s", "--super", default=False,
                    help="whether super or not")
parser.add_argument("-a", "--anum", default=8,
                    help="number of aspect classes")

def chars2idx(chars, dic):
    idxes = []
    for w in chars:
        if w in dic:
            idx = dic[w]
        else:
            idx = dic['UNK']
        idxes.append(idx)

    return idxes

def getclass(rate):
    if rate >= 14.0:
        cls = 2
    elif rate >= 8.0:
        cls = 1
    else:
        cls = 0
    return cls

def label2weight(labels,labelnum):
    labeldic = dict(zip(range(labelnum),[0]*labelnum))
    for label in labels:
        labeldic[label] += 1
    weights = []
    weight = [0.0] * labelnum
    for label in range(labelnum):
        if labeldic[label] != 0:
            weight[label] = 1.0
    weights.append(weight)
    for label in labels:
        weight = [0.0] * labelnum
        weight[label] = 1.0/float(labeldic[label])
        weights.append(weight)
    return weights

def main():

    dic = pickle.load(open("vocab.pickle",'rb'))
    udic = pickle.load(open("udic.pickle",'rb'))
    pdic = pickle.load(open("pdic.pickle",'rb'))

    args = parser.parse_args()
    rews = []
    
    fin = open(args.input, 'r')
    lines = fin.readlines()
    fin.close()
    
    asplabelnum = int(args.anum)
    max_sen = 0
    miss_count = 0
    x = 0
    #print(args.super == 'True')
    for line in lines:
        rewdic = json.loads(line.strip())
        if 'words' in rewdic:
            text = rewdic['words']
        else:
            text = rewdic['text']
        pid = rewdic['pid'] 
        uname = rewdic['uname'] 
        orrate = getclass(rewdic['orrate'])

        sensid = []
        lens = []
        sennum = 0
        flag = False
        if text == []:
            #print(text)
            continue
        for sen in text:
            if len(sen) == 0:
                flag = True
                miss_count += 1
                break
            sennum += 1
            idxes =  chars2idx(sen, dic)
            sensid.append(idxes)
            lens.append(len(sen))

        if flag == True:
            #print(flag)
            continue
        if uname in udic:
            usrid = udic[uname]
        else:
            usrid = 0
        if pid in pdic:
            prdid = pdic[pid]
        else:
            print(pid)
            continue
        #orrate = getclass(orrate)

        if sennum > max_sen:
            max_sen = sennum
        
        if 'asplabels' in rewdic and 'ratelabels' in rewdic:
            asplabels = rewdic['asplabels']
            ratelabels = rewdic['ratelabels']
            aspweights = label2weight(asplabels,asplabelnum)
            datamask = 1.0
            x += 1
        else:
            asplabels = [0]*sennum
            ratelabels = [0]*sennum
            aspweights = []
            aspweights.append([1.0] * asplabelnum) 
            aspweights += [[0.0] * asplabelnum]*sennum
            datamask = 0.0
        '''
        if args.super == 'True':
            asplabels = rewdic['asplabels']
            ratelabels = rewdic['ratelabels']
            aspweights = label2weight(asplabels,asplabelnum)
            datamask = 1.0 #unsuper or super
            #print(args.super)
        else:
            asplabels = [0]*sennum
            ratelabels = [0]*sennum
            aspweights = []
            aspweights.append([1.0] * asplabelnum) 
            aspweights += [[0.0] * asplabelnum]*sennum
            datamask = 0.0
            #print(args.super)
        '''
        # condition, target, labels, lens, sennum, mask
        if sensid == []:
            print('none text',text)
            #print(idxes)
        rews.append((usrid, prdid, orrate, sensid, asplabels, ratelabels, sennum, lens, datamask, aspweights))

    print (x)
    print ("max_sen: %d" % (max_sen))
    print ("data num: %d" % (len(rews)))
    #random.shuffle(rews)
    print ("saving dev poems to %s" % (args.binarized))
    output = open(args.binarized, 'wb')
    pickle.dump(rews, output, -1)
    output.close()


if __name__ == "__main__":
    main()
