"""
readcsv.py

Reads the .csv file(s) containing sentences for training the Naive-Bayes algorithm.
Created by Larry Davis III
"""

import csv


class ReadCsv:

    @staticmethod
    def readcsv(file):
        """
        Reads given .csv file and returns an array with its data. Columns are separated
        :param: file:
        :return: columnarray1, columnarray2, ...
        """

        list1 = []
        list2 = []

        with open(file, 'r', newline='') as csvfile:

            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            spamreader = csv.reader(csvfile, dialect)
            for column in spamreader:
                list1.append(column[0])
                list2.append(column[1])

        # print(list1, list2)

        return list1, list2




