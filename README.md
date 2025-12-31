# LEARNING LATENT DYNAMICS: A UNIFIED FRAMEWORK FOR WAVEFORM RECONSTRUCTION AND INVERSE MODELING

The goal of this work was to show whether transformers could be used to speed up tasks in traditional gravitational wave analysis, as they are able to capture the wave as a time series.
The desired tasks were parameter inference, next value prediction and missing value imputation.
A decoder-only transformer was trained for the first two tasks, while an encoder-only transformer was used for the imputation task.
A regression head was added as the final layer of the models, connected to the last output vector of the transformer.
To mimic actual gravitational waves, a combination of sine and cosine waves with differing frequencies was used.
The waves were created with fixed time steps and fixed length.

This work was created as part of a semestral work at FIT CTU under the supervision of Mr. Ippocratis Saltas.

## Tasks

### Parameter inference

This task involves predicting (inferring) the parameters that created / generated the wave, which is one of the most important tasks in gravitational wave analysis.
A combination of a sine wave (its frequency is our target parameter) and a fixed frequency cosine squared wave was used.
The trained model (with around 400k parameters) was able to reach an average MAE of 0.286 on the frequency of the sine wave (ranging from 85 to 115 Hz).

### Next value prediction

Similar to LLMs, a transformer could be used to continue a waveform, which could be used to warn of an incoming event (like a merger).
For this task, a more complicated signal consisting of three different sine waves was used.
The final model (1.5M parameters) was able to continue the wave for around 75 steps with decent precision, but after that the cascading error becomes too high.

### Missing value imputation

This task consists of filling in missing parts of a waveform.
A combination of two different sine waves was use as the data to train the model, with the missing parts emulated by replacing certain parts of the input with highly negative values. The model was then trained to output the entire wave with the missing parts filled in.
The resulting model (1.7M parameters) was able to fill in these missing parts with a median L1 average error on these parts of 0.0465. 

## Repository structure

Training and evaluation notebooks are in the root of the repository.
The trained models's weights can be found under the models folder.
To run the notebooks, you can install the requirements.txt and the notebook package.
The final paper can be found in report.pdf