#ifndef PERCEPTRON_PERCEPTRON_H
#define PERCEPTRON_PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>

#define SIMPLE_ACTIVATION 0 // If set to FALSE, then the sigmoid is used as an activation function

#if !SIMPLE_ACTIVATION
#define UNIPOLAR_ACTIVATION 1 // Whether to use the unipolar function or not
#define GRADIENT_DESCENT 1 // Whether or not to use gradient descent for weight update
#endif


class Perceptron {
    //Variables sector
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

    // Function declarations
public:
#if !SIMPLE_ACTIVATION

    Perceptron(double beta, double learning_rate, double acceptable_error); // Pass the required initial values
    double training_error(); // Output the error in training
    void update_weights(); // Update the weights using the provided learning rate
#else
    void update_weights(double actual_value, std::size_t data_index); // Update the weights in a simple way
#endif

    void train(); // Train the neuron with the provided dataset
    void classify(); // Classify the provided data set
    void weight_init(); // Initialize the weights with random numbers
};


#endif //PERCEPTRON_PERCEPTRON_H
