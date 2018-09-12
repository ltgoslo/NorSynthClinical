import sys
from collections import defaultdict


choice = None
path1, path2 = sys.argv[1], sys.argv[2]

taraka_files = ["example1.ann", "example2.ann", "example3.ann", "example4.ann", "example5.ann", "pal_090318.ann", "pal_230318.ann", "pal_300518.ann", "pal_310518.ann", "pal_initial.ann"]

lilja_files = ["pal_120618.ann", "pal_150618.ann"]

oe_files = ["example1.ann", "example2.ann", "example3.ann", "example4.ann", "example5.ann"]

if "taraka" in path1:
    choice = taraka_files
elif "lilja" in path1:
    choice = lilja_files
elif "oystein" in path1:
    choice = oe_files

gbl_ent_agree, gbl_left_spans, gbl_gold_spans = 0.0, 0.0, 0.0
nr_sents = 0.0
gbl_rel_agree, gbl_rel_left, gbl_rel_gold = 0.0, 0.0, 0.0

for fname in choice:
    file1, file2 = path1+fname, path2+fname
    nr_sents += len(open(file1[:-4]+".txt").readlines())
    
    print(file1, file2, nr_sents)
    

    entity_file1_d = defaultdict()
    entity_file2_d = defaultdict()

    rel_file1_d = defaultdict()
    rel_file2_d = defaultdict()

    file1_entity2span, file2_entity2span = defaultdict(), defaultdict()

    for line in sorted(open(file1, "r").readlines(), reverse=True):
        arr = line.strip().split("\t")
        if arr[0][0] == "T":
            arr_entity = arr[1].split(" ")
            entity_tag = arr_entity[0]
            entity_span = arr_entity[1]+" "+arr_entity[2]
            if entity_tag in ["NA", "X"]:
                continue
            entity_file1_d[entity_span] = [arr[-1], entity_tag]
            file1_entity2span[arr[0]] = entity_span
        elif arr[0][0] == "R":
    #        continue
            relation, left, right = arr[1].split(" ")
            left, right = sorted([left.split(":")[1], right.split(":")[1]])
            if left not in file1_entity2span or right not in file1_entity2span:
                continue
            left_span, right_span = file1_entity2span[left], file1_entity2span[right]
            rel_file1_d[left_span, right_span] = relation
            rel_file1_d[right_span, left_span] = relation

    for line in sorted(open(file2, "r").readlines(), reverse=True):
        arr = line.strip().split("\t")
        if arr[0][0] == "T":
            arr_entity = arr[1].split(" ")
            entity_tag = arr_entity[0]
            entity_span = arr_entity[1]+" "+arr_entity[2]
            if entity_tag in ["NA", "X"]:
                continue
            entity_file2_d[entity_span] = [arr[-1], entity_tag]
            file2_entity2span[arr[0]] = entity_span
        elif arr[0][0] == "R":
    #        continue
            relation, left, right = arr[1].split(" ")
            left, right = sorted([left.split(":")[1], right.split(":")[1]])
            left_span, right_span = file2_entity2span[left], file2_entity2span[right]
            rel_file2_d[left_span, right_span] = relation
            rel_file2_d[right_span, left_span] = relation        

    entity_agreement_1, entity_agreement_2 = 0.0, 0.0
    relation_agreement_1, relation_agreement_2 = 0.0, 0.0
    nr_spans_1, nr_spans_2 = 0.0, 0.0
    nr_relations_1, nr_relations_2 = 0.0, 0.0

    for k1, v1 in entity_file1_d.items():
        nr_spans_1 += 1.0
        if " " in v1[0]:
            nr_spans_1 += len(v1[0].split(" ")) -1.0
        if k1 in entity_file2_d:
            v2 = entity_file2_d[k1]
            if v1[1] == v2[1]:
                entity_agreement_1 += 1.0
                if " " in v1[0]:
                    entity_agreement_1 += len(v1[0].split(" ")) -1.0

    #    else:
    #        print(k1, v1)

    for k1, v1 in rel_file1_d.items():
        nr_relations_1 += 1.0
        if k1 in rel_file2_d:
            if rel_file2_d[k1] == v1:
                relation_agreement_1 += 1.0

    for k1, v1 in entity_file2_d.items():
        nr_spans_2 += 1.0
        if " " in v1[0]:
            nr_spans_2 += len(v1[0].split(" ")) -1.0
        if k1 in entity_file1_d:
            v2 = entity_file1_d[k1]
            if v1[1] == v2[1]:
                entity_agreement_2 += 1.0
                if " " in v1[0]:
                    entity_agreement_2 += len(v1[0].split(" ")) -1.0
    #    else:
    #        print(k1, v1)

    for k1, v1 in rel_file2_d.items():
        nr_relations_2 += 1.0
        if k1 in rel_file1_d:
            if rel_file1_d[k1] == v1:
                relation_agreement_2 += 1.0


    #print("Agreed spans ", span_agreement, nr_spans, round(float(span_agreement*100/nr_spans),3))
#    print("Agreed entities right gold", entity_agreement_1, nr_spans_1, nr_spans_2)
    #print("Agreed entities left gold", entity_agreement_2, nr_spans_2, round(float(entity_agreement_2*100/nr_spans_2),3))

#    print("Agreed relations right gold", relation_agreement_1, nr_relations_1)
    #print("Agreed relations left gold", relation_agreement_2, nr_relations_2, round(float(relation_agreement_2*100/nr_relations_2),3))

    gbl_ent_agree += entity_agreement_1
    gbl_left_spans += nr_spans_1
    gbl_gold_spans += nr_spans_2
    
    gbl_rel_agree += relation_agreement_1
    gbl_rel_left += nr_relations_1
    gbl_rel_gold += nr_relations_2
    
p_e = gbl_ent_agree/gbl_left_spans
r_e = gbl_ent_agree/gbl_gold_spans
f1_e = 2.0*p_e*r_e/(p_e+r_e)
print("Nr. of sentences ",  nr_sents)
print(round(p_e,3), round(r_e,3), round(f1_e,3))

p_rel = gbl_rel_agree/gbl_rel_left
r_rel = gbl_rel_agree/gbl_rel_gold
f1_rel = 2.0*p_rel*r_rel/(p_rel+r_rel)
print(round(p_rel,3), round(r_rel,3), round(f1_rel,3))


