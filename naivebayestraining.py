"""
naivebayestraining.py

Trains the Naive-Bayes algorithm on question creation. Uses sklearn version. Uses readcsv.py for training.
Created by Larry Davis
"""

from readcsv import ReadCsv
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB

# dumping .csv file into lists

yparam = [ReadCsv.readcsv
                   (r"C:\Users\elden\PycharmProjects\QuestionCreation\Question Creation Documents\questiondata.csv")]
xparam = [ReadCsv.readcsv
                   (r"C:\Users\elden\PycharmProjects\QuestionCreation\Question Creation Documents\questiondata.csv")]
print(xparam, yparam)

# Create model
model = GaussianNB()

# Train model
model.fit(xparam, yparam)



