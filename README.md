# Bayesian Logistic Regression Modelling for Classifying Back Pain Abnormality

Back pain is a health condition that affects millions of people globally. Unfortunately, determining the causes of back pain is challenging due to the structural complexity of the spine. However, modern technology allows us to quantify characteristics of an individual's spine. With Bayesian logistic regression modelling, it's possible to process spine data to diagnose back pain abnormalities. The benefits of producing this model are:

* Proactively inform a patient of potential future back pain and advise on preventative steps (e.g. keep lumbar at 180 degrees at all time)
* Diagnose patients with chronic existing back pain and prescribe treatment (pelvic joint inflammation medication)
* Understand key drivers behind a normal healthy back and an abnormal unhealthy back.

Using Bayesian logistic regression methodology, it is possible to capture the relationships between back pain abnormality and back features. There are several reasons for this choice over other statistical learning models:

* Given that the patient spine dataset is not large (310 data points), Bayesian methodology works elegantly by using expert opinion (priors) and updating it through the dataset (likelihood). The end result is a posterior distribution that incorporates the best of expert opinion and observed patient spine data.
* Using Bayesian methods enables the use of probability distributions and credible sets for modelling features (i.e. $\beta$ coefficients). This gives the advantage of understanding how confident Bayesian estimates are and how they are associated with back pain abnormalities (response variable).

### How to Run Code through Python

Please ensure you run the code with Python 3.6.8>=. For this project, I use Python 3.8.3. I also recommend using [Python 3 Virtual Environments](https://docs.python.org/3/library/venv.html) to install the dependencies in an isolated environment. This is optional.

The code below will install the requirements and then run the project code.

```
pip install -r requirements.txt
python bayes_project.py
```

Sample Output:
```
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

For your convenience, there is also a __reports/Bayes Project (With Visuals).html__ that let's you see the code and visuals through your web browser.


### Project Structure

- __bayes\_project.py__ - Python executable to run Bayesian logistic regression pymc3 sampler.
- __Bayes Project (With Visuals).ipynb__ - This file runs the bayes\_project.py code in a Jupyter notebook and also shows the visual seen in the report
- __reports/Bayes Project (With Visuals).html__ - This file is the HTML/web browser viewable version of the notebook code and the graphs shown in the report.
- __reports/report.pdf__ - Submitted PDF report about project background, data analysis, model formulation, and results
- __README.txt__ - Project documentation


### Troubleshooting

#### Mac (OSX)

It is possible you may run into an issue where Theano fails to run on OSX. This is not an issue with the application but rather installation of PyMC3 and Theano. Ensure:

* You have the developer tools installed via xcode which contain things like the C compiler
* If you run into Theano missing C headers, The below should resolve it: 
```
cd /Library/Developer/CommandLineTools/Packages/
open macOS_SDK_headers_for_macOS_10.14.pkg
```

#### Windows/Other

The application should run fine on Windows and Linux assuming installations are correct.
