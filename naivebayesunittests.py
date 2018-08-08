"""
naivebayesunittest.py

Unit tests for this implementation of the Naive-Bayes algorithm
Created by Larry Davis III
"""
from unittest import TestCase
from naivebayestraining import TrainNaiveBayes


class TestCsvReader(TestCase):

    def testSentenceList(self):
        self.assertIsNotNone(TrainNaiveBayes.readcsv())

    def testParser(self):
        pass

    def test


if __name__ == "__main__":
    TestCsvReader.testSentenceList()
