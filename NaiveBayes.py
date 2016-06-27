import numpy as np
import pandas as pd
import os

email_path = "/Users/dannyg/Desktop/Projects/NaiveBayesProject/Emails/"

email_files = [x for x in os.listdir(email_path)]

emails = []

for e in email_files:
	with open(email_path + e, 'r') as email:
		emails.append(email.read())

print emails[0]
		
