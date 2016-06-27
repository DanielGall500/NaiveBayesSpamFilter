import numpy as np
import pandas as pd
import os

email_path = "/Users/dannyg/Desktop/Projects/NaiveBayesProject/NaiveBayesSpamFilter/Emails/"

email_files = [x for x in os.listdir(email_path)]

emails = []
labels = []

filter_chars = [',', '?', '  ']

for e in email_files:
	with open(email_path + e, 'r') as email:
		text = [x for x in email.read().translate(None, '\n').split(' ')]

		for ch in filter_chars:
			text = [x.replace(ch, ' ') for x in text]

		lbl = text[-1]

		labels.append(lbl)

		del text[-1] #so it doesn't try use the label in predctions

		emails.append(text)

print emails[1]
print labels





















		
