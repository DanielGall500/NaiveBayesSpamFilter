import numpy as np
import pandas as pd
import os

email_path = "/Users/dannyg/Desktop/Projects/NaiveBayesProject/NaiveBayesSpamFilter/Emails/"

email_files = [x for x in os.listdir(email_path)]

emails = []
labels = [] #0 = spam, 1 = not spam

filter_chars = [',', '?', '!', '  ', '   ', '    ']

for e in email_files:
	with open(email_path + e, 'r') as email:
		text = [x for x in email.read().replace('\n', ' ')]

		for ch in filter_chars:
			text = [x.replace(ch, ' ') for x in text]

		text = ''.join(text).split()

		lbl = text[-1]
		labels.append(lbl)
		del text[-1] #so it doesn't try use the label in predctions

		emails.append(text) 

def create_frequency_table(texts, labels):
	freq_tbl = pd.DataFrame([])

	for t, lbl in zip(texts, labels):
		vocab = set(t)

		d = pd.Series({ v : t.count(v) for v in vocab})
		d['*class*'] = lbl

		freq_tbl = freq_tbl.append(d, ignore_index=True)

	freq_tbl = freq_tbl.fillna(0)

	return freq_tbl

del labels[0]
del emails[0] #TODO

def prob_of_classes(labels):
	probs = {}

	classes = set(labels)

	for c in classes:
		all_c = [x for x in labels if x == c]

		prob_class = float(len(all_c)) / float(len(labels))

		probs[c] = prob_class

	return probs

freq_tbl = create_frequency_table(emails, labels)

class_probs = prob_of_classes(labels)




























		
