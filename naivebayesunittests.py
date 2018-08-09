"""
naivebayesunittest.py

Unit tests for this implementation of the Naive-Bayes algorithm
Created by Larry Davis III
"""
from unittest import TestCase
from naivebayestraining import TrainNaiveBayes


class TestCsvReader(TestCase):

    def testSentenceList(self):
        global sentencelist
        global questionlist
        sentencelist, questionlist = TrainNaiveBayes.readcsv(r"C:\Users\elden\PycharmProjects\QuestionCreationNaiveBayes\QuestionCreation-Naive_Bayes\Question Creation Documents\questiondata.csv")
        self.assertIsNotNone(sentencelist)
        self.assertNotEqual(sentencelist, questionlist)
        self.assertEqual(len(sentencelist), len(questionlist))

    def testParser(self):
        parsedsentences, parsedquestions = TrainNaiveBayes.partsofspeech(sentencelist, questionlist, r"C:\Users\elden\PycharmProjects\sentence-classifier\config.json")
        self.assertIsInstance(parsedsentences[0], tuple, "Did not return tuples with parts of speech")

    def test(self):
        pass


if __name__ == "__main__":
    TestCsvReader.testSentenceList()
