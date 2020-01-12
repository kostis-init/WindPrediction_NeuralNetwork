# Wind Prediction using Neural Network

Predict wind intensity and experiment with a pre-trained neural network, using Keras API over TensorFlow.

## Prerequisites (Python 3.7)

* Numpy
* TensorFlow
* Keras API
* Pandas

## Usage

A. $ python3.7 predict.py -i [input_file]

B. $ python3.7 new_representation.py -i [input_file]

C. See [cluster's readme](cluster/README.md)

## Predict Results

We used [the pre_trained neural network](input_data/WindDenseNN.h5) to experiment and check how close are [the predicted values](output_data/predicted.csv) with the [actual ones](input_data/actual.csv). Check the results [here](output_data/predicted.csv) or see below:

* MAE: 0.05263347910583771
* MAPE: 37.70071941785548
* MSE: 0.004686631263251593

## Cluster Results

We performed clustering (Algorithm 212) for [initial nn representation](input_data/nn_representations.csv), [new representation](output_data/new_representation.csv) and [actual data](input_data/actual.csv). Check the results [here](cluster_results) or see below:

* 4 Clusters
    * Initial: Silhouette Evaluation = 0.222197, Time Elapsed = 1.8591
    * New: Silhouette Evaluation = 0.381512, Time Elapsed = 0.916453
    * Actual: Silhouette Evaluation = 0.433872, Time Elapsed = 0.662896

* 12 Clusters
    * Initial: Silhouette Evaluation = 0.200354, Time Elapsed = 24.7621
    * New: Silhouette Evaluation = 0.215712, Time Elapsed = 2.22797
    * Actual: Silhouette Evaluation = 0.26619 , Time Elapsed = 1.27526

## Conclusion

* Prediction: The mean values show that the predicted values, generated by the neural network that is given, are close enough to the actual ones.
* Clustering: The results show that the new representation, produced by executing only the first layer of the neural network, are better than the initial ones. In addition, the actual data results are better than initial and new as expected.

## Authors

* [Kostis Michailidis](https://github.com/kostismich7) (AM: 1115201500098)
* [Loukas Litsos](https://github.com/lkslts64) (AM: 1115201500082)
