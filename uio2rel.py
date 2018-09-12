from collections import defaultdict
import sys, random, glob
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import itertools as it

random.seed(1234)

seed = 1234

all_classes = ["No_Relation"]

def extract_relations():
    sent_id = 0
    relations = defaultdict(list)
    sentences = defaultdict()

    for f in sorted(glob.glob(sys.argv[1]+"/*.uio")):
        print(f)
        
        sent = None
        map_space_word = defaultdict(lambda: defaultdict())
        
        for line in open(f, "r"):
            if line.startswith("# sent_id =  "):
                sent_id += 1
                continue
            elif line.startswith("# text =  "):
#                print("# sent_id ", sent_id, line[10:-1], sep="\t")
                sent = line[10:-1]

                sentences[sent_id] = sent

                n_space = 1
                map_space_word[sent_id][0] = 0
                for i, ch in enumerate(list(sent)):
                    if ch == " ":

                        map_space_word[sent_id][i+1] = n_space
                        n_space += 1

            elif line == "\n":
                continue
            else:
                arr = line.strip().split("\t")
                rel_arr = [arr[1]]

                if arr[1] not in all_classes:
                    all_classes.append(arr[1])


                for e in arr[2:]:
                    tag, l, r = e.split(",")

                    rel_arr.append([tag, map_space_word[sent_id][int(l)]]) 
                    
                    
                    w = sent[int(l):int(r)]

                    if " " in w:
                        for j, ch in enumerate(list(w)):
                            if ch == " ":
                                rel_arr[-1].append(int(map_space_word[sent_id][int(l)+j+1]))

#                    print([tag]+[sent.split(" ")[x] for x in rel_arr[-1][1:]])
#                print(rel_arr, sent_id)
#                print([sent.split(" ")[y] for x in rel_arr[1:] for y in x[1:] ])
                relations[sent_id].append(rel_arr)

    return (relations, sentences)

def read_parse_file():
    parses = defaultdict()
    sent_id = None
    sent = []
    for line in open(sys.argv[2], "r"):
        if line.startswith("# sent_id"):
            sent_id = int(line.strip().split(" = ")[1])
        elif line == "\n":
            parses[sent_id] = sent
            sent = []
        else:
            arr = line.strip().split("\t")
            sent.append((arr[1], arr[3], arr[-3]))
    return parses

def read_predicted_entities(f):
    preds = defaultdict()
    sent_id = None
    sent = []
    for line in open(f, "r"):
        if line.startswith("# sent_id"):
            sent_id = int(line.strip().split(" = ")[1])
        elif line == "\n":
            preds[sent_id] = sent
            sent = []
        else:
            arr = line.strip().split("\t")
            sent.append((arr[0], arr[1]))
    return parses

def get_features(relation, sent, parse, pred_entity=None):
    feat_dict = {}

    start, end = relation[1], relation[2]

    if relation[1][1] > relation[2][1]:
        end, start = start, end

    feat_dict["entity1"] = start[0]
    feat_dict["entity2"] = end[0]

#    if not pred_entity:
#        feat_dict["entity1"] = pred_entity[start[0]][1]
#        feat_dict["entity2"] = pred_entity[end[0]][1]


#    feat_dict["temp_prevwrd1"], feat_dict["temp_nextwrd1"] = parse[start[1]-1][0], parse[start[1]+1][0]
#    feat_dict["temp_prevwrd2"], feat_dict["temp_nextwrd2"] = parse[end[1]-1][0], parse[end[1]+1][0]

    temp_deplabel, temp_pos, temp_word, temp_entity = [], [], [], []

#    for x, r in enumerate(start[1:]):
#        feat_dict["word1_"+str(x)] = parse[r][0]
#        feat_dict["deplabel1_"+str(x)] = parse[r][-1]
#        feat_dict["pos1_"+str(x)] = parse[r][1]
#        feat_dict["entity1_"+str(x)] = pred_entity[r][1]

#    for x, r in enumerate(end[1:]):
#        feat_dict["word2_"+str(x)] = parse[r][0]
#        feat_dict["deplabel2_"+str(x)] = parse[r][-1]
#        feat_dict["pos2_"+str(x)] = parse[r][1]
#        feat_dict["entity2_"+str(x)] = pred_entity[r][1]


    for r in start[1:]:
        temp_word.append(parse[r][0])
        temp_deplabel.append(parse[r][-1])
        temp_pos.append(parse[r][1])
        temp_entity.append(pred_entity[r][1])

    feat_dict["wrd1"] = " ".join(temp_word)
    feat_dict["pos1"] = " ".join(temp_pos)
    feat_dict["deplabel1"] = " ".join(temp_deplabel)
#    feat_dict["entity1"] = " ".join(temp_entity)

    temp_deplabel, temp_pos, temp_word, temp_entity = [], [], [], []

    for r in end[1:]:
        temp_deplabel.append(parse[r][-1])
        temp_pos.append(parse[r][1])
        temp_word.append(parse[r][0])

    feat_dict["wrd2"] = " ".join(temp_word)
    feat_dict["pos2"] = " ".join(temp_pos)
    feat_dict["deplabel2"] = " ".join(temp_deplabel)
#    feat_dict["entity2"] = " ".join(temp_entity)

    return feat_dict

def get_other_rels(rel_arr):
    x = defaultdict(lambda: defaultdict())
    other_rels = []
    for r in rel_arr:
        x[tuple(r[1])][tuple(r[2])] = r[0]
        x[tuple(r[2])][tuple(r[1])] = r[0]
    
    es = list(x.keys())

    for k1, k2 in it.combinations(es, r=2):
        if k2 not in x[k1]:
            other_rels.append(["No_Relation", list(k1), list(k2)])

    return other_rels

relations, sentences = extract_relations()
parses = read_parse_file()
sent_indexes = list(relations.keys())

#random.shuffle(sent_indexes)
#print(sent_indexes)

kf = KFold(n_splits=int(sys.argv[3]), shuffle=True, random_state=seed)

print(all_classes)

n_iter = 1

average_prf = []

pred_entities = read_predicted_entities("ner_predicted.txt")        

#avg_confusion_matrix = np.zeros((len(all_classes),len(all_classes)))

n_train, n_test = [], []

for train, test in kf.split(sent_indexes):
    train_feats, test_feats = [], []
    train_y, test_y = [], []

    print(len(train), len(test))
    print("Iteration {}".format(n_iter))
#    print(test)
    for idx in sent_indexes:
        relation_arr, sent, parse = relations[idx], sentences[idx], parses[idx]
        
#        aug_parse = [("START_2", "START_1")]+parse+[("END_1", "END_2")]

        aug_parse = parse

        other_rels = []
        other_rels = get_other_rels(relation_arr)

        for rel in relation_arr+other_rels:
#            if rel[0] == "Others":
#                print(rel)
            if idx in train:            
                train_y.append(all_classes.index(rel[0]))
                train_feats.append(get_features(rel, sent, aug_parse, pred_entity=pred_entities[idx]))
            else:
                test_y.append(all_classes.index(rel[0]))
                test_feats.append(get_features(rel, sent, aug_parse, pred_entity=pred_entities[idx]))

    v = DictVectorizer()
    train_x = v.fit_transform(train_feats)

    print("number of features {}".format(len(v.vocabulary_)))
    print("number of train instances {}".format(len(train_y)))
    print("number of test instances {}".format(len(test_y)))

    n_train.append(len(train_y))
    n_test.append(len(test_y))
    
    lin_clf = LinearSVC()
    lin_clf.fit(train_x, train_y)

    test_x = v.transform(test_feats)

    pred_y = lin_clf.predict(test_x)
    
    test_labels = [all_classes[i] for i in test_y]
    pred_labels = [all_classes[i] for i in pred_y]
    print(classification_report(test_labels, pred_labels, digits=3))
    print(confusion_matrix(test_labels, pred_labels))
#    avg_confusion_matrix += confusion_matrix(test_labels, pred_labels)
    n_iter += 1
    score_arr = precision_recall_fscore_support(test_labels, pred_labels, average='weighted')
    print(*score_arr[:-1])
    average_prf.append([score_arr[0], score_arr[1], score_arr[2]])

print("Average SVM P, R, F ",np.array(average_prf).mean(axis=0).round(3))
#print()
#print(avg_confusion_matrix/int(sys.argv[3]))
print("Average number of training instances ", np.mean(np.array(n_train)))
print("Average number of test instances ", np.mean(np.array(n_test)))
