import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

import numpy as np
data_file = sys.argv[1]


start_seq = ["START_2", "START_1"]
end_seq = [ "END_1", "END_2"]

def get_feats(words,  pos_tags, index, pred_tags):
    
    word, pos = words[index], pos_tags[index]
    
    feat_dict = {
                 "word":word, 
                 "pos":pos,
                 "prev_word":words[index-1],
                 "prev_tag":pos_tags[index-1],
                 "next_word":words[index+1],
                 "next_tag":pos_tags[index+1],
#                 "word_lower": word.lower(),
#                 "prefix1": word[0],
#                 "prefix2": word[0:2],
#                 "prefix3": word[0:3],
#                 "suffix1": word[-1],
#                 "suffix2": word[-2:],
#                 "suffix3": word[-3:],
                 "prev_word2":words[index-2],
                 "prev_tag2":pos_tags[index-2],
                 "next_word2":words[index+2],
                 "next_tag2":pos_tags[index+2],
                 "prev_iobtag1":pred_tags[-1],
                 "prev_iobtag2":pred_tags[-2],
#                 "word_bgp":words[index-1]+":"+words[index],
#                 "word_bgn":words[index+1]+":"+words[index],
#                 "pos_bgp":pos_tags[index-1]+":"+pos_tags[index],
#                 "pos_bgn":pos_tags[index+1]+":"+pos_tags[index],
                 }
    return feat_dict

def read_files(f):
    sents, sent, all_classes = [], [], []
    for line in open(f,  "r"):
        if line.startswith("#"):continue
        elif line.strip() == "":
            sents.append(sent)
            sent = []
        else:
            arr = line.strip().split("\t")
            sent.append((arr[1], arr[3], arr[-1]))
            if arr[-1] not in all_classes:
                all_classes.append(arr[-1])
    return (sents, all_classes)
            
    
sents, all_classes = read_files(data_file)

kf = KFold(n_splits=int(sys.argv[2]), shuffle=True, random_state=1234)
n_iter = 1
average_prf = []
baseline_prf = []

fw = open("ner_predicted.txt", "w")

for train, test in kf.split(list(range(len(sents)))):
    train_feats, test_feats = [], []
    train_y, test_y = [], []
    
    print(len(train), len(test))
    print("Iteration {}".format(n_iter))

    for idx in train:
        words, pos_tags, entity_tags = zip(*sents[idx])
        words = start_seq + list(words) + end_seq
        pos_tags = start_seq + list(pos_tags) + end_seq

        pred_tags = ["START_2", "START_1"]

        for index in range(len(entity_tags)):
            train_feats.append(get_feats(words, pos_tags, index, pred_tags))
            train_y.append(all_classes.index(entity_tags[index]))
            pred_tags.append(entity_tags[index])

    v = DictVectorizer()
    train_x = v.fit_transform(train_feats)

    print("number of features {}".format(len(v.vocabulary_)))
    print("number of train instances {}".format(len(train_y)))

    lin_clf = LinearSVC()
    lin_clf.fit(train_x, train_y)

    for idx in test:

        words, pos_tags, entity_tags = zip(*sents[idx])
        words = start_seq + list(words) + end_seq
        pos_tags = start_seq + list(pos_tags) + end_seq
        
        pred_tags = ["START_2", "START_1"]
        
        print("# sent_id = ", idx+1, file=fw)

        for index in range(len(entity_tags)):            
            test_y.append(all_classes.index(entity_tags[index]))
            
            current_feats = get_feats(words, pos_tags, index, pred_tags)
            test_feats.append(current_feats)
            pred_tags.append(all_classes[lin_clf.predict(v.transform(current_feats))[0]])
            print(words[index+2], pred_tags[-1], file=fw, sep="\t")    
        print(file=fw)

    print("number of test instances {}".format(len(test_y)))

    test_x = v.transform(test_feats)

    pred_y = lin_clf.predict(test_x)
    
    test_labels = [all_classes[i] for i in test_y]
    pred_labels = [all_classes[i] for i in pred_y]
    print(classification_report(test_labels, pred_labels, digits=3))
    print(confusion_matrix(test_labels, pred_labels))

    score_arr = precision_recall_fscore_support(test_labels, pred_labels, average='weighted')
    average_prf.append([score_arr[0], score_arr[1], score_arr[2]])

    baseline_prf.append(list(precision_recall_fscore_support(test_labels, ["O"]*len(test_labels), average='weighted')[:-1]))
    
    n_iter+=1

fw.close()

print("Average SVM P, R, F ",np.array(average_prf).mean(axis=0).round(3))

print("Average SVM P, R, F ",np.array(baseline_prf).mean(axis=0).round(3))


