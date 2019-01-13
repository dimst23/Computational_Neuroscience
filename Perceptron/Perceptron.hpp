#ifndef PERCEPTRON_PERCEPTRON_H
#define PERCEPTRON_PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>

#define SIMPLE_ACTIVATION 0 // If set to TRUE, then the sigmoid is used as an activation function
#define CLEAR_OUTPUT_ARRAY true // Used to clear the neuron output array

class Perceptron {
private:
    double sum; // Hold the weighted sum in accumulation

#if !SIMPLE_ACTIVATION
    double beta; // Beta value used as a parameter in the sigmoid activation function
    double learning_rate; // Learning rate used in weight update
    double acceptable_error; // Maximum acceptable error value for the training session
#endif

public:
    std::vector<std::vector<double>> input_data; // Data ready for classification or use in training
    std::vector<int> tags; // Actual values for the classification training
    std::vector<double> neuron_output; // Classified values
    std::vector<double> weights; // Hold the neuron weights

private:
    double activation(); // Activation function of the neuron

#if !SIMPLE_ACTIVATION
    void update_weights(); // Update the weights using the provided learning rate
#else
    void update_weights(double actual_value, std::size_t data_index); // Update the weights in a simple way
#endif

public:
#if !SIMPLE_ACTIVATION
    Perceptron(double beta, double learning_rate, double acceptable_error); // Pass the required initial values
#endif
    void train(); // Train the neuron with the provided dataset
    void classify(bool clear = false); // Classify the provided data set
    void weight_init(); // Initialize the weights with random numbers
};


#endif //PERCEPTRON_PERCEPTRON_H
