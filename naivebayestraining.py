"""
naivebayestraining.py

Trains the Naive-Bayes algorithm on question creation. Uses sklearn version. Uses readcsv.py for training.
Created by Larry Davis III
"""

from readcsv import ReadCsv
from stanfordcorenlp import StanfordCoreNLP
from utilities import load_configuration_data



# dumping .csv file into lists


class TrainNaiveBayes:

    @staticmethod
    def readcsv(file):
        """
        Reads given .csv file at file path

        :param file: path of needed file
        :return sentence_list: list of sentences in the first column of the .csv file
        :return question_list: list of questions in the second column of the .csv file
        """
        global sentence_list
        global question_list
        sentence_list, question_list = ReadCsv.readcsv(file)
        # print(sentence_list)
        return sentence_list, question_list

    @staticmethod
    def partsofspeech(list_of_sentences, list_of_questions, configpath):
        """
        Parses sentences and questions into partsofspeech, etc using Stanford's corenlp library

        :param list_of_sentences: list of sentences obtained by readcsv() function
        :param list_of_questions: list of questions obtained by readcsv() function
        :param configpath: config.json filepath, needed for corenlp library, described in its documentation
        :return parsed_sentence_list: List of tuples that contain each word in a sentence and its part of speech
        :return parsed_question_list: List of tuples that contain each word in a question and its part of speech
        """

        module_settings = load_configuration_data(configpath)

        nlp = StanfordCoreNLP(module_settings["CORE_NLP"])

        parsed_sentence_list = [nlp.pos_tag(sentence) for sentence in list_of_sentences]
        parsed_question_list = [nlp.pos_tag(question) for question in list_of_questions]

        print(parsed_question_list, parsed_sentence_list)
        return parsed_sentence_list, parsed_question_list


if __name__ == "__main__":
    TrainNaiveBayes.readcsv(r"C:\Users\elden\PycharmProjects\QuestionCreationNaiveBayes\QuestionCreation-Naive_Bayes\Question Creation Documents\questiondata.csv")
    TrainNaiveBayes.partsofspeech(sentence_list, question_list, r"C:\Users\elden\PycharmProjects\sentence-classifier\config.json")

