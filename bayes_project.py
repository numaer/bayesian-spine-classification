""" Bayesian Back Modelling

Using spine data, trains a Bayesian logistic regression model
and prints the model performance.

Numaer Zaker <nzaker3@gatech.edu>
IsYE 6420 - Bayesian Statistics - Fall 2020
---

References used in the creation of this project:

Dataset: https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset
PyMC3 Documentations: https://docs.pymc.io/api.html
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
    print("Beginning sampling progress for posteriors of our random variables..")
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
        We use the mean of the predictive posterior as the probability.
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

    return y_hat, y_hat_mean, y_hat_pred, prediction_results

if __name__ == '__main__':
    raw_data = load_data()
    X_train, X_test, y_train, y_test, train_data = split_data(raw_data)
    trace = bayesian_model(train_data)
    calculate_model_performance(trace, X_test, y_test)

    
"""
Sample Output:

$> python bayes_project.py
Load in and clean our data
Split dataset into training and testing (80/20)
Beginning sampling progress for posteriors of our random variables..
Auto-assigning NUTS sampler...
Initializing NUTS using adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [degree_spondylolisthesis, pelvic_radius, lumbar_lordosis_angle, pelvic_tilt, Intercept]
Sampling 4 chains, 0 divergences: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 8000/8000 [00:32<00:00, 244.29draws/s]
===== BAYESIAN LOGISTIC REGRESSION POSTERIOR STATISTICS =====
                            mean     sd  hpd_3%  hpd_97%  mcse_mean  mcse_sd  ess_mean  ess_sd  ess_bulk  ess_tail  r_hat
Intercept                 10.047  2.788   4.910   15.191      0.082    0.059    1156.0  1105.0    1171.0    1648.0    1.0
pelvic_tilt                0.114  0.034   0.057    0.182      0.001    0.001    2308.0  2190.0    2334.0    2515.0    1.0
lumbar_lordosis_angle     -0.075  0.018  -0.107   -0.039      0.000    0.000    1661.0  1661.0    1658.0    2117.0    1.0
pelvic_radius             -0.080  0.021  -0.120   -0.042      0.001    0.000    1201.0  1132.0    1222.0    1674.0    1.0
degree_spondylolisthesis   0.169  0.027   0.122    0.223      0.001    0.000    1592.0  1592.0    1569.0    2197.0    1.0
Testing model on test set...
===== MODEL PERFORMANCE ON TEST DATA =====
# Correctly Classified:  53
# Incorrectly Classified:  9
% Model Accuracy:  85.48


These results are discussed thoroughly in the paper. But at a high level the output shows:

* Degree spondylolisthesis is the strongest indicator of abnormal back
* Lumbar angle and pelvic radius are weaker indicators of a normal back
* Pelvic tilt is a indicator of an abnormal back
* Our model accurately classifies 85.48% of the test set data points.

"""