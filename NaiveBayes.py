import numpy as np
import pandas as pd
import os

email_path = "/Users/dannyg/Desktop/Projects/NaiveBayesProject/NaiveBayesSpamFilter/Emails/"

email_files = [x for x in os.listdir(email_path)]

to_filter = [',', '?', '!', '  ', '   ', '    ']

def parse_files(email_files, email_path):

	emails = []
	labels = [] #0 = spam, 1 = not spam

	for e in email_files:
		with open(email_path + e, 'r') as email:
			text = [x for x in email.read().lower().replace('\n', ' ')]

			for ch in to_filter:
				text = [x.replace(ch, ' ') for x in text]

			text = ''.join(text).split()

			lbl = text[-1]
			labels.append(lbl)
			del text[-1] #so it doesn't try use the label in predctions

			emails.append(text) 
	return emails, labels


class SpamFilter(object):

	def __init__(self, emails, to_filter=None):
		self.emails = emails
		self.to_filter = to_filter

	def create_frequency_table(self, texts=None, labels=None):
		freq_tbl = pd.DataFrame([])

		if not texts:
			texts = self.emails
			print 'not texts'

		for idx, t in enumerate(self.emails):
			vocab = set(t)

			d = pd.Series({ v : t.count(v) for v in vocab})

			if labels != None:
				d['*class*'] = labels[idx]

			freq_tbl = freq_tbl.append(d, ignore_index=True)

		return freq_tbl.fillna(0)

	def train(self, frequency_table):
		
		frequencies = frequency_table.iloc[:, 1:]
		labels = frequency_table.iloc[:, 0].values

		print frequencies

		vocab = list(frequencies.columns.values)[0]

		spam, nonspam = pd.DataFrame([]), pd.DataFrame([])

		for idx, row in frequencies.iterrows():
			print labels[idx]
			if labels[idx] == '1':
				spam = spam.append(row)
			else:
				nonspam = nonspam.append(row)

		nonspam_probs, spam_probs = {}, {}

		spam_word_count = sum([word for word in spam.sum()])
		nonspam_word_count = sum([word for word in nonspam.sum()])

		alpha = 1

		for word in vocab:
			print 'w: ', word
			print 'sp: ', spam
			word_occurences_spam = spam[word].sum()
			word_occurences_nonspam = nonspam[word].sum()

			bayesian_prob_spam = (word_occurences_spam + alpha) / (spam_word_count + len(vocab))
			bayesian_prob_nonspam = (word_occurences_nonspam + alpha) / (spam_word_count + len(vocab))

			nonspam_probs[word], spam_probs[word] = bayesian_prob_nonspam, bayesian_prob_spam

		self.nonspam_pr = nonspam_probs
		self.spam_pr = spam_probs

	def predict(self, text):

		text = [x for x in text.replace('\n', ' ')]

		for ch in self.to_filter:
			text = [x.replace(ch, ' ') for x in text]

		prsd_text = [''.join(text).split()]

		txt_table = self.create_frequency_table(texts=prsd_text)
		vocab = txt_table.columns.values

		spam_likelihood = 0
		nonspam_likelihood = 0

		for wrd in vocab:

			if wrd in self.spam_pr:
				spam_likelihood += self.spam_pr[wrd]

			if wrd in self.nonspam_pr:
				nonspam_likelihood += self.spam_pr[wrd]

		print 'Spam Likelihood: ', spam_likelihood
		print 'NonSpam Likelihood: ', nonspam_likelihood

		return int((spam_likelihood / nonspam_likelihood) >= 1) #Bayesian classifier


emails, labels = parse_files(email_files, email_path)

del labels[0]
del emails[0] #TODO

print emails

nb_spamfilter = SpamFilter(emails=emails, to_filter=to_filter)

email_tbl = nb_spamfilter.create_frequency_table(labels=labels)

nb_spamfilter.train(frequency_table=email_tbl)

eml_test = 'Hi Daniel johanne work for you regards'
pred = nb_spamfilter.predict(eml_test)

print pred

"""
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
"""




























		
