import os, math, collections, itertools, csv, re
import pickle
import nltk, nltk.classify.util, nltk.metrics
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.metrics.scores import precision, recall
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures

posData = []
negData = []

# with open('twitter_corpus.csv', 'rb') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         if row['Sentiment'] == '1':
#             posData.append(row['SentimentText'])
#         else:
#             negData.append(row['SentimentText'])
#
#
# f = open('positiveSentences.pickle', 'wb')
# g = open('negativeSentences.pickle', 'wb')
# pickle.dump(posData, f)
# pickle.dump(negData, g)
# f.close()
# g.close()



        #print(row['first_name'], row['last_name'])

# posData = []
# negData = []
f = open('positiveSentences.pickle', 'rb')
g = open('negativeSentences.pickle', 'rb')
posData = pickle.load(f)
negData = pickle.load(g)
f.close()
g.close()

def evaluate_features(feature_select):
    positiveFeatures = []
    negativeFeatures = []
    posCutoff = int(math.floor(len(posData) * 3 / 4))
    negCutoff = int(math.floor(len(negData) * 3 / 4))
    for i in posData:
        #print i
        #posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords = [feature_select(i), 'positive']
        # print posWords
        positiveFeatures.append(posWords)
    for i in negData:
        # negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords = [feature_select(i), 'negative']
        negativeFeatures.append(negWords)

    trainFeatures = positiveFeatures[:posCutoff] + negativeFeatures[:negCutoff]
    testFeatures = positiveFeatures[posCutoff:] + negativeFeatures[negCutoff:]

    # trainFeatures = posData[:posCutoff] + negData[:negCutoff]
    # testFeatures = posData[posCutoff:] + negData[negCutoff:]

    # trains a Naive Bayes Classifier
    print "begin training the classifier"
    # def train_features_generator():
    #     for items in trainFeatures:
    #         yield items
    # here dmcn
    classifier = NaiveBayesClassifier.train(trainFeatures)
    x = open('huge_classifier.pickle', 'wb')
    pickle.dump(classifier, x)
    x.close()
# creates a feature selection mechanism that uses all words
# w
def make_full_dict(words):
    return dict([(word, True) for word in words])


# tries using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)