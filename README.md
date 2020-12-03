# Bayesian Logistic Regression Modelling for Back Pain Abnormality

Numaer Zaker \<nzaker3@gatech.edu\><br/>
IsYE 6420 - Bayesian Statistics - Fall 2020 Project<br/>
Georgia Institute of Technology<br/>

### How to Run Code through Python

Please ensure you run the code with python 3.6.8>. For this project, I use Python 3.8.3. The code below will install the requirements and then run the project code

```
pip install -r requirements.txt
python bayes_project.py
```

Sample Output:
```
$> python bayes_project.py
Load in and clean our data
Split dataset into training and testing (80/20)
Beginning sampling progress for posteriors of beta random variables..
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
```

### How to Run Code Through Jupyter Notebook

Please install [Jupyter notebook here](https://jupyter.org/install). Once you run have jupyter notebook running:

1. Open the __Bayes Project (With Visuals).ipynb__ notebook file
2. Ensure you are running the notebook in the same directory as __bayes_project.py__ (the notebook depends on this file)
3. Clear all outputs and run all the cells.

The output will:

* Generate the analysis graphs shown in the report
* Builds the bayesian logistic regression model
* Show the accuracy and respective results plot.

For your convenience, there is also a __Bayes Project (With Visuals).html__ that let's you see the code and visuals without Jupyter Notebook.


### Project Structure

- __bayes\_project.py__ - Python executable to run Bayesian logistic regression pymc3 sampler.
- __Bayes Project (With Visuals).ipynb__ - This file runs the bayes\_project.py code in a Jupyter notebook and also shows the visual seen in the report
- __Bayes Project (With Visuals).html__ - This file is the HTML/viewable version of the notebook code and the graphs shown in the report.
