import csv
import re

sentences = set([])
for i,row in enumerate(csv.reader(open("4_-_annotation_template_phrase_1000.csv"))):
    if i:
        sentences.add(row[0])

candidates = [["sentence", "candidate", "label"]]
for s in sentences:
    for p in re.split("[、。]", s):
        if p:
            candidates.append([s, p, "PROFILE"])

csv.writer(open("5_-_annotation_template_phrase_1000.csv", "w")).writerows(candidates)