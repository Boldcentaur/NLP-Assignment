# models.py

from sentiment_data import *
from utils import *

from collections import Counter, defaultdict
import numpy as np
import nltk
from nltk.corpus import stopwords


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer
    
    def word_occurance(self, train_exs):
        return None

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        preprocessed_sentence = self.preprocessing(sentence)
        counter = Counter(preprocessed_sentence)
        for feature in counter.keys():
            self.indexer.add_and_get_index(feature, add_to_indexer)
        return counter
    
    def preprocessing(self, sentence: List[str]) -> List[str]:
        preprocessed_sentence = [''.join(filter(str.isalpha, word)) for word in sentence]
        return [word.lower() for word in preprocessed_sentence if word.lower() not in self.stop_words and word]

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def word_occurance(self, train_exs):
        return None

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        preprocessed_sentence = self.preprocessing(sentence)
        bigram_sentence = self.combine_word(preprocessed_sentence)
        counter = Counter(bigram_sentence)
        for feature in counter.keys():
            self.indexer.add_and_get_index(feature, add_to_indexer)
        return counter
    
    def preprocessing(self, sentence: List[str]) -> List[str]:
        preprocessed_sentence = [''.join(filter(str.isalpha, word)) for word in sentence if word]
        return [word.lower() for word in preprocessed_sentence]

    def combine_word(self, sentence: List[str]) -> List[str]:
        return [sentence[i] + '|' + sentence[i+1] for i in range(len(sentence) - 1)]

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.total_training_documents = 0
        self.word_count = defaultdict(int)

    def get_indexer(self):
        return self.indexer

    def word_occurance(self, train_exs: List[SentimentExample]) -> None:
        for train_ex in train_exs:
            preprocessed_sentence = self.preprocessing(train_ex.words)
            for word in set(preprocessed_sentence):
                self.word_count[word] += 1
            self.total_training_documents += 1

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        preprocessed_sentence = self.preprocessing(sentence)
        counter = Counter(preprocessed_sentence)
        tf_idf = {}
        for word in set(preprocessed_sentence):
            tf_score = counter[word] / len(preprocessed_sentence)
            try:
                idf_score = np.log(self.total_training_documents/self.word_count[word])
            except:
                idf_score = np.log(self.total_training_documents)
            tf_idf[word] = tf_score * idf_score
        for feature in tf_idf.keys():
            self.indexer.add_and_get_index(feature, add_to_indexer)
        return tf_idf

    def preprocessing(self, sentence: List[str]) -> List[str]:
        preprocessed_sentence = [''.join(filter(str.isalpha, word)) for word in sentence]
        return [word.lower() for word in preprocessed_sentence if word]



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_vector, featurizer):
        self.weight_vector = weight_vector
        self.featurizer = featurizer
    
    def predict(self, sentence: List[str]) -> int:
        decision_boundary = 0
        counter = self.featurizer.extract_features(sentence, False)
        for word in counter.keys():
            index = self.featurizer.get_indexer().index_of(word)
            decision_boundary += counter[word] * self.weight_vector[index]
        if decision_boundary > 0:
            return 1
        else:
            return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_vector, featurizer):
        self.weight_vector = weight_vector
        self.featurizer = featurizer
    
    def predict(self, sentence: List[str]) -> int:
        intermediate = 0
        counter = self.featurizer.extract_features(sentence, False)
        for word in counter.keys():
            index = self.featurizer.get_indexer().index_of(word)
            intermediate += counter[word] * self.weight_vector[index]
        decision_boundary = logistic(intermediate)
        if decision_boundary > 0.5:
            return 1
        else:
            return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    nltk.download('stopwords')
    learning_rate = 0.1
    weight_vector = np.zeros(100000)
    num_epoch = 100
    np.random.seed(2021)

    for i in range(num_epoch):
        permutation_list = np.random.permutation(len(train_exs))
        for j in permutation_list:
            counter = feat_extractor.extract_features(train_exs[j].words, True)
            decision_boundary = 0
            indices = []
            for word in counter.keys():
                index = feat_extractor.get_indexer().index_of(word)
                decision_boundary += counter[word] * weight_vector[index]
                indices.append(index)
            if decision_boundary > 0 and train_exs[j].label == 0:
                for index in indices:
                    word = feat_extractor.get_indexer().get_object(index)
                    weight_vector[index] -= learning_rate * counter[word]
            elif decision_boundary <= 0 and train_exs[j].label == 1:
                for index in indices:
                    word = feat_extractor.get_indexer().get_object(index)
                    weight_vector[index] += learning_rate * counter[word]
    
    return PerceptronClassifier(weight_vector, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    nltk.download('stopwords')
    learning_rate = 0.01
    weight_vector = np.zeros(200000)
    num_epoch = 35
    np.random.seed(2021)

    feat_extractor.word_occurance(train_exs)

    for i in range(num_epoch):
        permutation_list = np.random.permutation(len(train_exs))
        for j in permutation_list:
            counter = feat_extractor.extract_features(train_exs[j].words, True)
            intermediate = 0
            indices = []
            for word in counter.keys():
                index = feat_extractor.get_indexer().index_of(word)
                intermediate += counter[word] * weight_vector[index]
                indices.append(index)
            decision_boundary = logistic(intermediate)
            if decision_boundary > 0.5 and train_exs[j].label == 0:
                for index in indices:
                    word = feat_extractor.get_indexer().get_object(index)
                    weight_vector[index] -= (learning_rate * counter[word] * (1 - (1 - logistic(intermediate))))
            elif decision_boundary <= 0.5 and train_exs[j].label == 1:
                for index in indices:
                    word = feat_extractor.get_indexer().get_object(index)
                    weight_vector[index] += (learning_rate * counter[word] * (1 - logistic(intermediate)))
    # print(feat_extractor.get_indexer().objs_to_ints)
    return LogisticRegressionClassifier(weight_vector, feat_extractor)


def logistic(number):
    return np.exp(number)/(1 + np.exp(number))


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model