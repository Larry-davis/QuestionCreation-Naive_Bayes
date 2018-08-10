"""
naivebayesunittest.py

Unit tests for this implementation of the Naive-Bayes algorithm
Created by Larry Davis III
"""
from unittest import TestCase
from naivebayestraining import TrainNaiveBayes


class TestCsvReader(TestCase):

    def run_unittests(self):
        """
        runs unittests for naivebayestraining.py
        :param self:
        :return:
        """

        # unittests for the function readcsv()

        sentencelist, questionlist = TrainNaiveBayes.readcsv(r"C:\Users\elden\PycharmProjects\QuestionCreationNaiveBayes\QuestionCreation-Naive_Bayes\Question Creation Documents\questiondata.csv")
        self.assertIsNotNone(sentencelist)
        self.assertNotEqual(sentencelist, questionlist)
        self.assertEqual(len(sentencelist), len(questionlist))

        # unittests for the function partsofspeech()

        parsedsentences, parsedquestions = TrainNaiveBayes.partsofspeech(sentencelist, questionlist, r"C:\Users\elden\PycharmProjects\sentence-classifier\config.json")
        self.assertIsInstance(parsedsentences[0], tuple, "Did not return tuples with parts of speech")
        self.assertIn('NN', parsedsentences, "Did not parse correctly")
        self.assertIn('NN', parsedquestions, "Did not parse correctly")

        # unittests for the function fit_and_predict()
        self.assertIsInstance(TrainNaiveBayes.fit_and_predict(sentencelist, questionlist), str, "Did not return string")

if __name__ == "__main__":
    runner = TestCsvReader()
    runner.run_unittests()
