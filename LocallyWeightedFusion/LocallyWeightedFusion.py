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
sys.path.insert(0,'/home/lacava/code/ellyn/ellyn')
from ellyn import ellyn
from ._version import __version__
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

class LocallyWeightedFusion(BaseEstimator):

    """Multifactor Dimensionality Reduction (LocallyWeightedFusion) for feature construction in machine learning"""

    def __init__(self, base_estimator = ellyn(), n_estimators = 10, threshold=0.05):
        """Sets up the LocallyWeightedFusion algorithm

        Parameters
        ----------
        d: ('mahalanobis' or 'euclidean')
            Type of distance calculation to use

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

    def fit(self, features, labels):
        """Constructs the LocallyWeightedFusion from the provided training data

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        None

        """
        #
        if isinstance(self.base_estimator, ellyn):
            # get estimators as final population
            self.base_estimator.fit(features,labels)
            hof = self.base_estimator.get_hof()
            for m in hof:
                self.estimators.append(ellyn(self.base_estimator.__dict__))
                self.estimators.self.best_estimator_ = m
        else:

        # define neighborhood threshold for each sample as 5% of the range of the sample
        self.R = [self.threshold*np.range(x) for x in features]
        self.X_train = features
        self.Y_train = labels

    def predict(self, Q):
        """Predict class outputs for an unlabelled feature set"""

        peers = self._neighborhood(Q)
        # evaluate models on peers
        predictions = np.array([est.predict(peers) for est in self.estimators])
        # calculate mean absolute error
        mae = np.sum(np.abs(predictions-self.Y_train))/len(peers)
        # calculate mean error (bias)
        if bias;
            me = np.sum(predictions-self.Y_train)/len(peers)
        else:
            me = 0
        # get weights
        w = 1/mae
        # make weighted prediction
        y_pred = np.sum(w*(predictions-me))/np.sum(w)

        return y_pred

    def fit_predict(self, features, classes):
        """Convenience function that fits the provided data then predicts the class labels
        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of true class labels

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        self.fit(features, classes)
        return self.predict(features)

    def score(self, features, classes, scoring_function=accuracy_score, **scoring_function_kwargs):
        """Estimates the accuracy of the predictions from the constructed feature

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to predict from
        classes: array-like {n_samples}
            List of true class labels

        Returns
        -------
        accuracy_score: float
            The estimated accuracy based on the constructed feature

        """
        if not self.mu:
            raise ValueError('The LocallyWeightedFusion model must be fit before score() can be called')

        return scoring_function(classes, self.predict(features), **scoring_function_kwargs)

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
        return [x for i,x in enumerate(self.X_train) if np.sum((x-Q)**2) < self.R[i] ]

def main():
    """Main function that is called when LocallyWeightedFusion is run on the command line"""
    parser = argparse.ArgumentParser(description='LocallyWeightedFusion for classification based on distance measure in feature space.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to perform LocallyWeightedFusion on; ensure that the class label column is labeled as "class".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-d', action='store', dest='D', default='mahalanobis',choices = ['mahalanobis','euclidean'],
                        type=str, help='Distance metric to use.')

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
    # training_indices, testing_indices = train_test_split(input_data.index,
    #                                                      stratify=input_data['label'].values,
    #                                                      train_size=0.75,
    #                                                      test_size=0.25,
    #                                                      random_state=RANDOM_STATE)
    #
    # training_features = input_data.loc[training_indices].drop('label', axis=1).values
    # training_classes = input_data.loc[training_indices, 'label'].values
    #
    # testing_features = input_data.loc[testing_indices].drop('label', axis=1).values
    # testing_classes = input_data.loc[testing_indices, 'label'].values

    # Run and evaluate LocallyWeightedFusion on the training and testing data
    lwf = LocallyWeightedFusion(d = args.D)
    # dc.fit(training_features, training_classes)
    lwf.fit(input_data.drop('label',axis=1).values, input_data['label'].values)
    print(lwf.score(input_data.drop('label',axis=1).values, input_data['label'].values))

    # if args.VERBOSITY >= 1:
    #     print('\nTraining accuracy: {}'.format(dc.score(training_features, training_classes)))
    #     print('Holdout accuracy: {}'.format(dc.score(testing_features, testing_classes)))



if __name__ == '__main__':
    main()
