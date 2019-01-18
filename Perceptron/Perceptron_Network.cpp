#include "Perceptron_Network.hpp"

Perceptron_Network::Perceptron_Network(double beta, double learning_rate, double acceptable_error) {
    Perceptron_Network::beta = beta;
    Perceptron_Network::learning_rate = learning_rate;
    Perceptron_Network::acceptable_error = acceptable_error;
}

void Perceptron_Network::create_network() {
    if (input_data.empty() or tags.empty()) {
        throw std::invalid_argument("No input data or data tags provided for the network.\n"
                                    "Please enter data and/or data tags to the network.");
    } else if (input_data.size() != tags.size()) {
        throw std::invalid_argument("Input data and tags matrix sizes do not match.");
    }

    network_size = static_cast<int>(std::ceil(input_data.size() / 2.0)); // Get the required network size

    for (std::size_t i = 0; i < network_size; i++) {
        std::vector<int> neuron_tags; // Store temporary tags

        for (auto const& tag : tags) {
            neuron_tags.push_back(tag.at(i)); // Save the tags for each neuron
        }

        Perceptron perc(beta, learning_rate, acceptable_error); // Create the neuron in the network
        perc.input_data = input_data; // Give the input data to neuron
        perc.tags = neuron_tags; // Provide the classification tags to the neuron
        perc.weight_init(); // Initialize the weights on start
        neurons.push_back(perc); // Save the neuron into the array
    }
}

void Perceptron_Network::train_network() {
    if (neurons.empty()) {
        throw std::invalid_argument("There is no neural network.\n"
                                    "Please create the neural network first!");
    }

    for (auto& neuron : neurons) {
        neuron.train(); // Train each individual neuron
    }
}

void Perceptron_Network::net_classify(bool clear) {
    if (neurons.empty()) {
        throw std::invalid_argument("There is no neural network.\n"
                                    "Please create the neural network first!");
    }

    if (clear) {
        network_output.clear(); // Clear any leftover data in the output array
        network_output.resize(input_data.size()); // Preallocate the output sub arrays
    }

    for (auto& neuron : neurons) {
        std::size_t index = 0; // Index for the array elements
        neuron.classify(CLEAR_OUTPUT_ARRAY); // Classify the data for each neuron

        for (auto const& output : neuron.neuron_output) {
            network_output[index].push_back(output); // Save the classified value in the vector array
            index++; // Increment for the next iteration
        }
    }
}

