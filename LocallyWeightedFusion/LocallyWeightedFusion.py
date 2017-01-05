# -*- coding: utf-8 -*-

"""
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:


The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import sys
sys.path.insert(0,'/media/bill/data/Dropbox/PostDoc/code/ellyn/ellyn')
from ellyn import ellyn

from sklearn.linear_model import LassoLarsCV, LogisticRegression
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from ._version import __version__
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import copy
import pdb

class LocallyWeightedFusion(BaseEstimator):

    """Multifactor Dimensionality Reduction (LocallyWeightedFusion) for feature construction in machine learning"""

    def __init__(self, base_estimator = ellyn(selection='epsilon_lexicase',g=100,islands=True), n_estimators = 10, threshold=0.05, bias = True):
        """Sets up the LocallyWeightedFusion algorithm

        Parameters
        ----------
        base_estimator: ml method to use to generate models in the ensemble
        n_estimators: number of models (currently overridden in ellyn by population size)
        threshold: distance threshold for choosing neighbors of query points for prediction
        bias: include bias in weighting

        Returns
        -------
        None

        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.R = []
        self.X_train = np.empty([1])
        self.Y_train = np.empty([1])
        self.bias = bias

    def fit(self, features, labels):
        """Constructs the LocallyWeightedFusion from the provided training data

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        labels: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        None

        """
        self.estimators = []
        if isinstance(self.base_estimator, ellyn):
            self.base_estimator.return_pop = True
            # get estimators as final population
            self.base_estimator.fit(features,labels)
            # get unique models based on programs
            hof = [list(m) for m in set(tuple(m) for m in self.base_estimator.hof)]
            hof = hof[:self.n_estimators]
            for m in hof:
                self.estimators.append(copy.deepcopy(self.base_estimator))
                self.estimators[-1].best_estimator_ = m
        else:
            for i in np.arange(self.n_estimators):

                self.estimators.append(copy.deepcopy(self.base_estimator))
        # define neighborhood threshold for each sample as 5% of the range of the sample
        self.R = [self.threshold*(np.max(x)-np.min(x)) for x in features]
        self.X_train = features
        self.Y_train = labels.reshape(-1,1)
        self.Y_est = np.array([est.predict(self.X_train) for est in self.estimators]).transpose()
        # filter down to models with unique outputs
        # all_outs = np.array([est.predict(self.X_train) for est in self.estimators])
        # self.Y_est = np.vstack({tuple(row) for row in all_outs}).transpose()

    def predict(self, Q):
        """Predict class outputs for an unlabelled feature set"""
        y_pred = np.empty((Q.shape[0]))
        for i,q in enumerate(Q):
            loc,peers = self._neighborhood(q)
            # evaluate models on peers
            if loc: # if there are training points near the query point
                # calculate mean absolute error
                mae = np.sum(np.abs(self.Y_est[loc]-self.Y_train[loc]),axis=0)/len(peers)
                # calculate mean error (bias)
                if self.bias:
                    me = np.sum(self.Y_est[loc]-self.Y_train[loc],axis=0)/len(peers)
                else:
                    me = 0
                # get weight
                w = 1/mae
                w[np.isinf(w)] = np.max(w[~np.isinf(w)])*2
            else: # if no neighbors are found, define equal weights for models
                mae = 1
                me = 0
                w = np.ones((len(self.estimators),))

            # get model predictions
            predictions = np.array([est.predict(q.reshape(1,-1)) for est in self.estimators]).transpose()
            # make weighted prediction
            # pdb.set_trace()
            y_pred[i] = np.sum(w*(predictions-me))/np.sum(w)
            if np.isnan(y_pred[i]):
                pdb.set_trace()

        return y_pred

    def fit_predict(self, features, labels):
        """Convenience function that fits the provided data then predicts the class labels
        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        labels: array-like {n_samples}
            List of true class labels

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        self.fit(features, labels)
        return self.predict(features)

    def score(self, features, labels, scoring_function=mean_squared_error, **scoring_function_kwargs):
        """Estimates the accuracy of the predictions from the constructed feature

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to predict from
        labels: array-like {n_samples}
            List of true class labels

        Returns
        -------
        accuracy_score: float
            The estimated accuracy based on the constructed feature

        """
        if not self.R:
            raise ValueError('The LocallyWeightedFusion model must be fit before score() can be called')
        # tmp = self.predict(features)

        return scoring_function(labels, self.predict(features), **scoring_function_kwargs)

    def get_params(self, deep=None):
        """Get parameters for this estimator

        This function is necessary for LocallyWeightedFusion to work as a drop-in feature constructor in,
        e.g., sklearn.cross_validation.cross_val_score

        Parameters
        ----------
        deep: unused
            Only implemented to maintain interface for sklearn

        Returns
        -------
        params: mapping of string to any
            Parameter names mapped to their values
        """
        return self.params

    def _neighborhood(self,Q):
        """return neighborhood of Q, which is the set of training features near it."""

        tmp = [(i,x) for i,x in enumerate(self.X_train) if np.sum((x-Q)**2) < self.R[i] ]
        loc = [i for i,x in tmp]
        peers = np.array([x for i,x in tmp])


        return loc,peers

def main():
    """Main function that is called when LocallyWeightedFusion is run on the command line"""
    parser = argparse.ArgumentParser(description='LocallyWeightedFusion for classification based on distance measure in feature space.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to perform LocallyWeightedFusion on; ensure that the class label column is labeled as "class".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-est', action='store', dest='BASE_ESTIMATOR', default='ellyn',choices = ['ellyn','lasso'],
                        type=str, help='base estimator to use.')

    parser.add_argument('-n', action='store', dest='N_ESTIMATORS', default=10,
                        type=int, help='Number of estimators in ensemble.')

    parser.add_argument('-t', action='store', dest='THRESHOLD', default=0.05,
                        type=float, help='Number of estimators in ensemble.')

    parser.add_argument('--no_bias', action='store_false', dest='BIAS', default=True,
                        help='Do not include bias term in local weighting of ensemble.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1, choices=[0, 1, 2],
                        type=int, help='How much information LocallyWeightedFusion communicates while it is running: 0 = none, 1 = minimal, 2 = all.')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE', default=0,
                        type=int, help='Random state for train/test split.')

    parser.add_argument('--version', action='version', version='LocallyWeightedFusion {version}'.format(version=__version__),
                        help='Show LocallyWeightedFusion\'s version number and exit.')

    args = parser.parse_args()

    if args.VERBOSITY >= 2:
        print('\nLocallyWeightedFusion settings:')
        for arg in sorted(args.__dict__):
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
        print('')

    input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Label': 'label'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE if args.RANDOM_STATE > 0 else None
    #
    training_indices, testing_indices = train_test_split(input_data.index,
                                                        #  stratify=input_data['label'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=RANDOM_STATE )
    #
    training_features = input_data.loc[training_indices].drop('label', axis=1).values
    training_labels = input_data.loc[training_indices, 'label'].values
    #
    testing_features = input_data.loc[testing_indices].drop('label', axis=1).values
    testing_labels = input_data.loc[testing_indices, 'label'].values
    # Run and evaluate LocallyWeightedFusion on the training and testing data
    ml_dict = {
        'ellyn': ellyn(selection='epsilon_lexicase',popsize=1000,g=1000,islands=True),
        'lasso': LassoLarsCV(),
        'svr': SVR(),
        'lsvr': LinearSVR(),
        'lr': LogisticRegression(solver='sag'),
        'svc': SVC(),
        'lsvc': LinearSVC(),
        'rfc': RandomForestClassifier(),
        'rfr': RandomForestRegressor(),
        'dtc': DecisionTreeClassifier(),
        'dtr': DecisionTreeRegressor(),
        'knc': KNeighborsClassifier(),
        'knr': KNeighborsRegressor(),
        None: None
    }
    lwf = LocallyWeightedFusion(base_estimator=ml_dict[args.BASE_ESTIMATOR],
                                n_estimators = args.N_ESTIMATORS,
                                threshold=args.THRESHOLD,bias=args.BIAS)
    lwf.fit(training_features, training_labels)
    # lwf.fit(input_data.drop('label',axis=1).values, input_data['label'].values)
    # print(lwf.score(input_data.drop('label',axis=1).values, input_data['label'].values))

    if args.VERBOSITY >= 1:
         print('\nTraining accuracy: {}'.format(lwf.score(training_features, training_labels)))
         print('Holdout accuracy: {}'.format(lwf.score(testing_features, testing_labels)))



if __name__ == '__main__':
    main()
