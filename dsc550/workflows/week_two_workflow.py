import luigi
import numpy as np
import pandas as pd
import re
import warnings
import gc
import pickle
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class WeekTwoReusable():
    filename = ''
    target_txt = ''
    df = pd.DataFrame()

    def set_df(self, type):
        if type == 'continuous':
            self.filename = '../../data/source/reddit/controversial-comments.jsonl'
        full_df = pd.read_json(self.filename, lines=True)
        self.df = full_df.sample(1000)

    def get_corpus(self, type):
        self.set_df(type)
        df = self.df
        comments_only = df['txt']
        corpus = comments_only.tolist()
        return corpus

    def get_model(self, type, corpus, classifier):
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform(corpus)
        features = vector.toarray()
        if type == 'continuous':
            self.target_txt = 'con'

        target = self.df[self.target_txt]
        features_train, features_test, target_train, target_test = \
            train_test_split(features, target, test_size=0.25)
        """
            This next part is a total hack because I could not figure out how to get
            the two vectors and the model out from the same Task. They are all required
            and Luigi is not meant to output multiple files. This was the best I could do.
            I repeat this hack (to unpickle) in the next task. The model part is right and
            is the atomic output from this Task.
            
            The point is, I need features and target for the accuracy task, which is after
            this one.
        """
        features_pckl = open('../../data/interim/cont_features.pkl', 'wb')
        pickle.dump(features, features_pckl)
        target_pckl = open('../../data/interim/cont_target.pkl', 'wb')
        pickle.dump(target, target_pckl)

        model = classifier.fit(features_train, target_train)
        return model

class ContinuousCorpus(luigi.Task):
    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(path='../../data/processed/continuous_corpus.pkl')

    def run(self):
        wk2_reuse = WeekTwoReusable()
        corpus = wk2_reuse.get_corpus('continuous')
        pkl_file = open(self.output().path, 'wb')
        pickle.dump(corpus, pkl_file)
        pkl_file.close

class ContinuousLogisticL1Model(luigi.Task):
    def requires(self):
        return ContinuousCorpus()

    def output(self):
        return luigi.LocalTarget(path='../../models/continuous_model.pkl')

    def run(self):
        classifier = LogisticRegression(random_state=0, penalty='l1', solver='liblinear')
        pkl_in_file = open(self.input().path, 'rb')
        corpus = pickle.load(pkl_in_file)

        wk2_reuse = WeekTwoReusable()
        wk2_reuse.set_df('continuous')
        model = wk2_reuse.get_model('continuous', corpus, classifier)

        pkl_out_file = open(self.output().path, 'wb')
        pickle.dump(model, pkl_out_file)

class ContinuousLogisticL1Predict(luigi.Task):
    def requires(self):
        return ContinuousLogisticL1Model()

    def output(self):
        return luigi.LocalTarget(path='../../reports/logistic_l1.pdf')

    def run(self):
        pkl_file = open(self.input().path, 'rb')
        model = pickle.load(pkl_file)

        """
            Picking up my hack from earlier...
        """
        features_pckl = open('../../data/interim/cont_features.pkl', 'rb')
        features = pickle.load(features_pckl)
        target_pckl = open('../../data/interim/cont_target.pkl','rb')
        target = pickle.load(target_pckl)

        features_train, features_test, target_train, target_test = \
            train_test_split(features, target, test_size=0.25)

        test_predictions = model.predict(features_test)
        train_predictions = model.predict(features_train)

        accuracy_test = accuracy_score(target_test, test_predictions)
        accuracy_train = accuracy_score(target_train, train_predictions)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 10)
        pdf.multi_cell(400, 5, 'TRAIN Accuracy:' + str(accuracy_train) + '\n' +
            'TEST Accuracy:' + str(accuracy_test))
        pdf.output(self.output().path, 'F')

if __name__ == '__main__':
    luigi.run()
