"""
naivebayesunittest.py

Unit tests for this implementation of the Naive-Bayes algorithm
Created by Larry Davis III
"""
from unittest import TestCase
from naivebayestraining import TrainNaiveBayes


class TestCsvReader(TestCase):

    def test_sentence_list(self):
        """
        Tests the readcsv() function of naivebayestraining.py
        :return:
        """
        sentencelist, questionlist = TrainNaiveBayes.readcsv(r"C:\Users\elden\PycharmProjects\QuestionCreationNaiveBayes\QuestionCreation-Naive_Bayes\Question Creation Documents\questiondata.csv")
        self.assertIsNotNone(sentencelist)
        self.assertNotEqual(sentencelist, questionlist)
        self.assertEqual(len(sentencelist), len(questionlist))

        return sentencelist, questionlist

    def test_parser(self, sentencelist, questionlist):
        """
        Tests the partsofspeech() function of naivebayestraining.py
        :return:
        """
        parsedsentences, parsedquestions = TrainNaiveBayes.partsofspeech(sentencelist, questionlist, r"C:\Users\elden\PycharmProjects\sentence-classifier\config.json")
        self.assertIsInstance(parsedsentences[0], tuple, "Did not return tuples with parts of speech")
        self.assertIn('NN', parsedsentences)
        self.assertIn('NN', parsedquestions)

    def test(self):
        pass


if __name__ == "__main__":
    TestCsvReader.test_sentence_list()
    TestCsvReader.test_parser()
