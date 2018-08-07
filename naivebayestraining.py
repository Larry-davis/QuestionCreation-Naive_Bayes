"""
naivebayestraining.py

Trains the Naive-Bayes algorithm on question creation. Uses sklearn version. Uses readcsv.py for training.
Created by Larry Davis III
"""

from readcsv import ReadCsv
import numpy as np


# dumping .csv file into lists


class TrainNaiveBayes:

    @staticmethod
    def readcsv():
        sentencelist = [ReadCsv.readcsv
                        (
                            r"C:\Users\elden\PycharmProjects\QuestionCreation\Question Creation "
                            r"Documents\questiondata.csv")]
        questionlist = [ReadCsv.readcsv
                        (
                            r"C:\Users\elden\PycharmProjects\QuestionCreation\Question Creation "
                            r"Documents\questiondata.csv")]
        print(questionlist, sentencelist)
        return sentencelist, questionlist



