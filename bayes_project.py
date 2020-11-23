""" Bayesian Back Modelling

Light weight version of project without visuals
and simply trains Bayesian regression model on the entire dataset

Numaer Zaker <nzaker3@gatech.edu>
"""

import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.model_selection import train_test_split


DATASET_PATH = './dataset_spine.csv'
SELECTED_FEATURES = ['pelvic_tilt', 
        'lumbar_lordosis_angle',
        'pelvic_radius',
        'degree_spondylolisthesis']

spine_features = ['pelvic_incidence', 
        'pelvic_tilt', 
        'lumbar_lordosis_angle',
        'sacral_slope', 
        'pelvic_radius',
        'degree_spondylolisthesis', 
        'pelvic_slope',
        'direct_tilt', 
        'thoracic_slope', 
        'cervical_tilt',
        'sacrum_angle', 
        'scoliosis_slope']

def load_data():
    print("Load in and clean our data")
    raw_data = pd.read_csv(DATASET_PATH,
            header=None,
            skiprows=[0],
            names=spine_features +
            ['back_status','notes'])

    raw_data['binary_back_status'] = raw_data['back_status'].apply(lambda x: 1 if (x == 'Abnormal') else 0)
    return raw_data


def split_data(raw_data):
    print("Split dataset into training and testing (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(raw_data[spine_features], raw_data['binary_back_status'], test_size=0.20, random_state=1337)
    train_df = pd.concat([X_train, y_train], axis=1)

    return X_train, X_test, y_train, y_test, train_df



def bayesian_model(train_df):
    """ Build our Bayesian logistic regression model with specifications:

        * Use non informative prior N(0, 10^-6),
        * NUTs sampler, converges faster than Gibbs
        * Run for 1k samples
    """
    print("Beginning sampling progress for posteriors of beta random variables.. (if it gets stuck hit enter to wake up process)")
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula('binary_back_status ~ pelvic_tilt + lumbar_lordosis_angle + pelvic_radius + degree_spondylolisthesis',
                train_df[['binary_back_status'] + SELECTED_FEATURES],
                family=pm.glm.families.Binomial())
        trace = pm.sample(1000, tune=1000, init='adapt_diag')

    print("===== BAYESIAN LOGISTIC REGRESSION POSTERIOR STATISTICS =====")
    print(pm.summary(trace))
    return trace


def predict_spine_probability(trace, x):
        """ Calculate the probability of a single data point as a posterior predictive distribution
        We take the mean to give us the best single point estimate
        """

        x = x.values
        pelvic_tilt, lumbar_lordosis_angle, pelvic_radius, degree_spondylolisthesis = x[0], x[1], x[2], x[3]
        shape = np.broadcast(pelvic_tilt, lumbar_lordosis_angle, pelvic_radius, degree_spondylolisthesis).shape
        x_norm = np.asarray([np.broadcast_to(x, shape) for x in [pelvic_tilt, lumbar_lordosis_angle, pelvic_radius, degree_spondylolisthesis]])
        return 1 / (1 + np.exp(-(trace['Intercept'] + trace['pelvic_tilt']*x_norm[0] + trace['lumbar_lordosis_angle']*x_norm[1] + trace['pelvic_radius']*x_norm[2] + trace['degree_spondylolisthesis']*x_norm[3])))

def calculate_model_performance(trace, X_test, y_test):
    print("Testing model on test set...")
    prediction_wrapper = lambda x: predict_spine_probability(trace, x)
    y_hat = X_test[SELECTED_FEATURES].apply(prediction_wrapper, axis=1)
    y_hat_mean = y_hat.apply(lambda x: x.mean())
    y_hat_pred = y_hat_mean.apply(lambda x: 1 if x > 0.5 else 0)
    prediction_results = np.unique((y_hat_pred == y_test), return_counts=True)

    print("===== MODEL PERFORMANCE ON TEST DATA =====")
    print("# Correctly Classified: ", prediction_results[1][1])
    print("# Incorrectly Classified: ", prediction_results[1][0])
    print("% Model Accuracy: ", 100*round(prediction_results[1][1] / (prediction_results[1][1] + prediction_results[1][0]),4))

if __name__ == '__main__':
    raw_data = load_data()
    X_train, X_test, y_train, y_test, train_data = split_data(raw_data)
    trace = bayesian_model(train_data)
    calculate_model_performance(trace, X_test, y_test)
