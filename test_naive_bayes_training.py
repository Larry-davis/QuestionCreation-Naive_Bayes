"""
test_naive_bayes_training.py

Unit tests for this implementation of the Naive-Bayes algorithm
Created by Larry Davis III
"""
from unittest import TestCase
import naive_bayes_training as nbt


class TestCsvReader(TestCase):

    self.sentence_list = []
    self.question_list = []

    def test_read_csv(self):
        """
        runs unittests for naive_bayes_training.py
        :param self:
        :return:
        """

        # unittests for the function read_csv()

        sentencelist, questionlist = nbt.read_csv(r"C:\Users\elden\PycharmProjects\QuestionCreationNaiveBayes\QuestionCreation-Naive_Bayes\Question Creation Documents\questiondata.csv")
        self.assertIsNotNone(sentencelist)
        self.assertNotEqual(sentencelist, questionlist)
        self.assertEqual(len(sentencelist), len(questionlist))

    def test_parts_of_speech(self):
        """
        runs unittests for parts of speech.py
        :param self:
        :return:
        """
        parsedsentences, parsedquestions = nbt.parts_of_speech(sentencelist, questionlist, r"C:\Users\elden\PycharmProjects\sentence-classifier\config.json")
        self.assertIsInstance(parsedsentences[0], tuple, "Did not return tuples with parts of speech")
        self.assertIn('NN', parsedsentences, "Did not parse correctly")
        self.assertIn('NN', parsedquestions, "Did not parse correctly")

    def test_fit_and_predict(self):
        """
        runs unittests for parts of speech.py
        :param self:
        :return:
        """
        # unittests for the function fit_and_predict()
        
        self.assertIsInstance(nbt.fit_and_predict(self.sentencelist, self.questionlist), str, "Did not return string")
        self.assert

