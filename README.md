# Using transformers for gravitational waves

The goal of this work was to show whether transformers could be used to speed up tasks in traditional gravitational wave analysis, as they are able to capture the wave as a time series.
The desired tasks were parameter inference, next value prediction and missing value imputation.
A model was created and trained for each of these tasks, showing that the architecture could be used to perform them, obtaining better results if a larger model was trained.

Training notebooks can be seen in training/.
Evaluation of the trained models is in evaluation/, while the models' weights are in models/.
The final report / paper can be found in report.pdf