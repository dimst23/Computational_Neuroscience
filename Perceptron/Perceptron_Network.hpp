#ifndef LAYERED_NETWORK_PERCEPTRON_NETWORK_H
#define LAYERED_NETWORK_PERCEPTRON_NETWORK_H

#include "Perceptron.hpp"

class Perceptron_Network {
    // Variable declaration section
private:
    std::vector<Perceptron> neurons; // Hold the layer neurons
    int network_size; // Save the neural network size (number of neurons)

    double beta; // Beta value used as a parameter in the sigmoid activation function
    double learning_rate; // Learning rate used in weight update
    double acceptable_error; // Maximum acceptable error value for the training session

public:
    std::vector<std::vector<double>> input_data; // Data ready for classification or use in training
    std::vector<std::vector<int>> tags; // Actual values for the classification training
    std::vector<std::vector<double >> network_output; // Classified values

    // Function declaration section
private:

public:
    Perceptron_Network(double beta, double learning_rate, double acceptable_error); // Pass the required initial values
    void create_network(); // Create the neural network based on the input data
    void train_network(); // Train the neural network
    void net_classify(bool clear = false); // Classify the input data using the network
};


#endif //LAYERED_NETWORK_PERCEPTRON_NETWORK_H
