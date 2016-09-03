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


# print(row['first_name'], row['last_name'])

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
        # print i
        # posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords = [feature_select(i), 'positive']
        # print posWords
        positiveFeatures.append(posWords)
    for j in negData:
        # negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords = [feature_select(j), 'negative']
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
    x = open('huge_classifier.pickle', 'rb')
    classifier = pickle.load(x)
    x.close()
    # classifier = NaiveBayesClassifier.train(trainFeatures)
    # pickle.dump(classifier, x)
    # x.close()
    # initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    # pickle.dump(classifier, x)
    # x.close()
    # puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    # prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
    # classifier.show_most_informative_features(10)
# creates a feature selection mechanism that uses all words


def make_full_dict(words):
    return dict([(word, True) for word in words])


# tries using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)

#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
    #creates lists of all positive and negative words
    positiveFeatures = []
    negativeFeatures = []
    for i in posData:
        #print i
        #posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        # posWords = [feature_select(i), 'positive']
        # print posWords
        positiveFeatures.append(i)
    for j in negData:
        # negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        # negWords = [feature_select(j), 'negative']
        negativeFeatures.append(j)
    posWords = list(itertools.chain(*positiveFeatures))
    negWords = list(itertools.chain(*negativeFeatures))

    # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in positiveFeatures:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negativeFeatures:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1

    # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    #builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# finds word scores
word_scores = create_word_scores()


# finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


# creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


# numbers of features to select
numbers_to_test = [10, 100, 1000, 10000, 15000]
# tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print 'evaluating best %d word features' % num
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)