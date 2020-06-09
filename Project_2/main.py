import time
import os
import math

dictionary = dict()
prob = dict()
total_words = dict()

def count_words(path, docs):
    #Counting words and building dictionary
    clas_counts = dict()
    clas_total = 0
    for doc in docs:
            with open(path+'/'+doc,'r') as d:
                for line in d:
                    words = line.lower().split()
                    for word in words:
                        word = word.strip('() \'".,?:-')
                        if word != '' and word not in dictionary:
                            dictionary[word] = 1
                            clas_counts[word] = 1
                            clas_total += 1
                        elif word != '':
                            clas_counts.setdefault(word, 0)
                            dictionary[word] += 1
                            clas_counts[word] += 1
                            clas_total += 1
    return clas_counts, clas_total

def  naive_bayes_train():
    classes = os.listdir("train_data/")  #reading all folders in directory
    print ('------- Counting and building vocabulary -------')
    for clas in classes:
        path, direc, docs = next(os.walk("train_data/"+clas)) #separating files from folders
        prob[clas], total_words[clas] = count_words(path, docs)

    #Removing less frequent words with frequency less than 3.
    mark_del = []
    for word in dictionary:
        if dictionary[word] < 3:
            mark_del.append(word)
    for word in mark_del:
        del dictionary[word]
    print ('\n----- Dictionary of words has been created. -----\n')

    #Calculating probabilities for each class and all its words.
    for clas in classes:
        print ('Class being trained:', clas)
        total_count = total_words[clas]+len(dictionary)
        for word in dictionary:
            if word in prob[clas]:
                count = prob[clas][word]
            else:
                count = 1
            prob[clas][word] = float(count+1)/total_count
    print ('----- Training complete -----\n')


def calc_clas_prob(path, classes, doc):
    #Calculate probabilities for words in a document belonging to a particular class
    test_output = dict()
    for clas in classes:
        test_output[clas] = 0
    with open(path+'/'+doc, 'r') as doc:
        for line in doc:
            words = line.lower().split()
            for word in words:
                word = word.strip('() \'".,?:-')
                if word in dictionary:
                    for clas in classes:
                        test_output[clas] += math.log(prob[clas][word])
    #taking class with the highest probability
    max_prob = -1000000000
    for clas in test_output:
        if test_output[clas] > max_prob:
            max_prob = test_output[clas]
            output_clas = clas
    return output_clas


def naive_bayes_test():
    classes = os.listdir("20_newsgroups/")
    total_accuracy = 0
    for clas in classes:
        print ('Class being tested:', clas)
        correct = 0
        total = 0
        path, direc, docs = next(os.walk("20_newsgroups/" + clas))
        for doc in docs:
            total += 1
            test_output = calc_clas_prob(path, classes, doc)
            if test_output == clas:
                correct += 1
        #calculated classifier's accuracy for the class
        accuracy = float(correct)/total*100
        print('Accuracy:', accuracy, '%\n')
        total_accuracy += accuracy
    print ('----- Testing complete -----\n')
    print ('Accuracy of NB classifier is:', total_accuracy/20, '%\n')

start_time = time.time()
naive_bayes_train()
naive_bayes_test()
print("Total time taken to execute the code is: %s seconds." % (time.time() - start_time))