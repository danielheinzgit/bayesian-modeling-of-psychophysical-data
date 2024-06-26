# Cognitive Science Experiment: Bayesian Modelling of Target Detection in Visual Search and Crowding

This repository contains the code and materials for a cognitive science experiment investigating the effects of shape similarity and object numbers on target detection in visual search and crowding paradigms. The experiment was conducted using a controlled environment and PsychoPy software for stimulus presentation, response collection, and data storage.

## Table of Contents

1. Introduction
2. Experiment Design
3. Data
4. Models
5. References

# Introduction

The experiment aimed to examine the effects of varying shape similarity and object numbers on target detection in visual search and crowding paradigms. Ten participants from the Technical University of Darmstadt were involved in the study. The experiment was conducted in a controlled environment using a high-resolution monitor and PsychoPy software.

# Experiment Design

Visual stimuli consisted of multiple objects randomly positioned within an annulus, displayed against a grey background. Object shapes were generated by using a GAN with varied similarity levels controlled using ShapeComp (Morgenstern et al., 2021) as well as variations in the number of objects on display. After each trial, participants were asked whether one of the objects had a shape that differed from all other obejcts.

# Data

We have collected 14800 data points. Among other variables, each row in the table contains such information as 'target_present', 'response_yes', 'distance_setting', 'number_of_objects'.

# Models

Data analysis employed signal detection theory to compute sensitivity $ d' $ and criterion $ \lambda $, while the dataset was modeled using Bayesian linear regression, connecting linear regression to signal detection theory (Vuorre, 2017). Partial pooling across participants was introduced to improve the model.

# References

Morgenstern, Y., Hartmann, F., Schmidt, F., Tiedemann, H., Prokott, E., Maiello, G., & Fleming, R. W. (2021). An image-computable model of human visual shape similarity. PLoS computational biology, 17(6), e1008981.

Vuorre, Matti. 2017. “Bayesian Estimation of Signal Detection Models.” October 9, 2017. https://vuorre.com/posts/2017-10-09-bayesian-estimation-of-signal-detection-theory-models.
