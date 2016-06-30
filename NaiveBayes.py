import numpy as np
import pandas as pd
import os

email_path = "/Users/dannyg/Desktop/Projects/NaiveBayesProject/NaiveBayesSpamFilter/Emails/"

email_files = [x for x in os.listdir(email_path)]

filter_chars = [',', '?', '!', '  ', '   ', '    ']

def parse_files(email_files, email_path):

	emails = []
	labels = [] #0 = spam, 1 = not spam

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
	return emails, labels

def parse_text(text):

	text = [x for x in text.replace('\n', ' ')]

	for ch in filter_chars:
		text = [x.replace(ch, ' ') for x in text]

	text = ''.join(text).split()

	return [text]



def create_frequency_table(texts, labels=None, parse=False):
	freq_tbl = pd.DataFrame([])

	for idx, t in enumerate(texts):
		vocab = set(t)

		d = pd.Series({ v : t.count(v) for v in vocab})

		if labels != None:
			d['*class*'] = labels[idx]

		freq_tbl = freq_tbl.append(d, ignore_index=True)

	freq_tbl = freq_tbl.fillna(0)

	return freq_tbl

def prob_of_classes(labels):
	probs = {}
	classes = set(labels)

	for c in classes:
		all_c = [x for x in labels if x == c]

		prob_class = float(len(all_c)) / float(len(labels))

		probs[c] = prob_class

	return probs

def train(frequency_table):

	
	frequencies = frequency_table.iloc[:, 1:]
	labels = frequency_table.iloc[:, 0].values

	vocab = list(frequencies.columns.values)

	spam, nonspam = pd.DataFrame([]), pd.DataFrame([])

	for idx, row in frequencies.iterrows():
		if labels[idx] == '1':
			spam = spam.append(row)
		else:
			nonspam = nonspam.append(row)

	nonspam_probs, spam_probs = {}, {}

	spam_word_count = sum([word for word in spam.sum()])
	nonspam_word_count = sum([word for word in nonspam.sum()])

	alpha = 1

	for word in vocab:
		word_occurences_spam = spam[word].sum()
		word_occurences_nonspam = nonspam[word].sum()

		bayesian_prob_spam = (word_occurences_spam + alpha) / (spam_word_count + len(vocab))
		bayesian_prob_nonspam = (word_occurences_nonspam + alpha) / (spam_word_count + len(vocab))

		nonspam_probs[word], spam_probs[word] = bayesian_prob_nonspam, bayesian_prob_spam

	return nonspam_probs, spam_probs

def predict(text, nb_nonspam, nb_spam):
	prsd_text = parse_text(text)

	txt_table = create_frequency_table(prsd_text)
	vocab = txt_table.columns.values

	spam_likelihood = 0
	nonspam_likelihood = 0

	for wrd in vocab:
		print wrd
		if wrd in nb_spam:
			print 'Spam: ', nb_spam[wrd]
			spam_likelihood += nb_spam[wrd]
		if wrd in nb_nonspam:
			print 'NonSpam: ', nb_nonspam[wrd]
			nonspam_likelihood += nb_nonspam[wrd]

	print 'Spam Likelihood: ', spam_likelihood
	print 'NonSpam Likelihood: ', nonspam_likelihood

	return int(spam_likelihood > nonspam_likelihood)


emails, labels = parse_files(email_files, email_path)

del labels[0]
del emails[0] #TODO

freq_tbl = create_frequency_table(emails, labels)

class_probs = prob_of_classes(labels)

nb_nonspam, nb_spam = train(freq_tbl)

print nb_nonspam 
print nb_spam

eml_test = 'dear paddy'

print predict(eml_test, nb_nonspam, nb_spam)

import matplotlib.pyplot as plt

def jitter(array, scale=0.1):
	jitter = scale * (max(array) - min(array))
	return array + np.random.randn(len(array)) * jitter

x = nb_nonspam.values()
y = nb_spam.values()
n = nb_nonspam.keys()#vocab
print n

plt.scatter(jitter(x), jitter(y))

for i, text in enumerate(n):
	print text
	plt.annotate(text, (x[i], y[i]))

plt.show()





























		
