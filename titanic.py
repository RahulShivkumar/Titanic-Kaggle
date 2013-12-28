#!/usr/bin/env python
import re
import os
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from optparse import OptionParser

#
#  TO-DO Create a model to use age and fare directly. These two could help
#

#Create a directory/file if it doesn't exist
def mkdirIfNecessary(fileName):
    if not os.path.exists(os.path.dirname(fileName)):
        os.makedirs(os.path.dirname(fileName))

#Convert Gender into an integer to feed
def _convertGender(gender):
    if gender == 'female':
        gender = 1
    else:
        gender = 0
    return gender

#Pull out the title alone - Rev, Dr, Mr etc
def _getTitle(name):
    for word in name.split():
        if word.endswith('.'):
            title = word
    return title

#Convert the title into an int by hashing it
def _titleHash(title,gender):
    hashedTitle = ord(title[0]) + len(title) + gender
    return hashedTitle

#Convert the location into an integer
def _convertLocation(location):
    if location == 'S':
        location = 0
    else:
        if location == 'C':
            location = 1
        else:
            location = 2
    return location

#Take the cabin number and covert it into a code
def _convertCabin(cabinNum):
    if len(cabinNum) == 0:
        cabinCode = 0
    else:
        cabinCode = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", cabinNum)
        cabinCode = ord(cabinCode[0])
    return cabinCode

#Pull out the department from their ticket number. If this isn't present assume it to be zero
def _getDeptCode(ticket):
    deptName = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", ticket)
    if len(deptName) == 0:
        deptName = 'none'
    deptCode = ord(deptName[0]) + len(deptName)
    return deptCode


if __name__ == '__main__':
    p = OptionParser()
    p.add_option("-T", "--trainFile", action="store", dest="trainFile", help="Location of the Training File")
    p.add_option("-t", "--testFile", action="store", dest="testFile", help="Location of the Test File")
    p.add_option("-r", "--resultFile", action="store", dest="resultFile", help="Location of the Result File that will be created")

    opts, args = p.parse_args()

    print 'Reading Training File located at ' + str(opts.trainFile)
    csv_file_object = csv.reader(open(opts.trainFile, 'rb'))
    header = csv_file_object.next()
    data=[]

    for row in csv_file_object:
        #Convert all the rows into required format
        row[4] = _convertGender(row[4])
        title = _getTitle(row[3])
        row[3] = _titleHash(title,row[4])
        row[8] = _getDeptCode(row[8])
        row[10] = _convertCabin(row[10])
        row[11] = _convertLocation(row[11])
        data.append(row)
    data = np.array(data)

    result = data[0::,1].astype(np.float)
    features = data[0::,[2,3,4,8,10,11]].astype(np.float)
    fare = data[0::,9].astype(np.float)


    #Our Forest is ready to Rock N Roll
    forest = RandomForestClassifier(n_estimators = 1000)
    forest = forest.fit(features,result)

    print 'Forest is Trained, ready to Rock & Roll'

    #open the test file and our result file
    result_file_object = csv.reader(open(opts.testFile, 'rb'))
    mkdirIfNecessary(opts.resultFile)
    open_file_object = csv.writer(open(opts.resultFile, "wb"))
    result_file_object.next()

    #Write out header
    resultRow = []
    resultRow.append('Survived')
    resultRow.append('PassengerId')
    open_file_object.writerow(resultRow)

    print 'Beginning Prediction using Test File located at ' + str(opts.testFile)

    for row in result_file_object:
        resultRow = []
        unknownRow = []

        #Convert all the rows into required format
        row[3] = _convertGender(row[3])
        title = _getTitle(row[2])
        row[2] = _titleHash(title,row[3])
        row[7] = _getDeptCode(row[7])
        #if(row[8] == ''):
            #row[8] = np.mean(fare)
        row[9] = _convertCabin(row[9])
        row[10] = _convertLocation(row[10])

        #Set Up data to send into our predictor
        unknownRow.append(row[1])
        unknownRow.append(row[2])
        unknownRow.append(row[3])
        unknownRow.append(row[7])
        unknownRow.append(row[9])
        unknownRow.append(row[10])
        unknownRow = np.array(unknownRow)
        unknownRow.astype(np.float)

        #Set Up result row to send into csv writer
        resultRow.append(int(forest.predict(unknownRow)))
        resultRow.append(row[0])
        print "Id: " + str(resultRow[1]) +'\t' + "Survival Status: " + str(resultRow[0])

        #write our result out
        open_file_object.writerow(resultRow)

    print 'Done, required file is located at ' + str(opts.resultFile)