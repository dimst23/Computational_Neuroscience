#include "Perceptron.hpp"

#if !SIMPLE_ACTIVATION

Perceptron::Perceptron(double beta, double learning_rate, double acceptable_error) {
    Perceptron::beta = beta;
    Perceptron::learning_rate = learning_rate;
    Perceptron::acceptable_error = acceptable_error;
}

#endif

double Perceptron::activation() {
#if SIMPLE_ACTIVATION
    if (sum > 0.0)
        return 1.0;
    else
        return -1.0;
#else

#if UNIPOLAR_ACTIVATION
    return 1.0 / (1.0 + exp(-Perceptron::beta * Perceptron::sum));
#else
    return (1.0 - exp(-Perceptron::beta * Perceptron::sum)) / (1.0 + exp(-Perceptron::beta * Perceptron::sum));
#endif

#endif
}

void Perceptron::classify() {
    // Check if parameters are valid
    if (input_data.empty() or weights.empty()) {
        throw std::invalid_argument("No input data or weights provided.\n"
                                    "Please enter data and/or initialize the weights.");
    } else if (weights.size() != (input_data.at(0).size() + 1)) {
        throw std::invalid_argument("Input data and weight matrix sizes do not match.");
    }

    if (neuron_output.capacity() == 0) {
        neuron_output.resize(input_data.size()); // Allocate array space, if there is no space
    }

    std::size_t current_data_id = 0; // Index to save the output
    for (auto const &data : input_data) {
        sum = 0.0; // Rest the sum for the next iteration
        for (std::size_t j = 0; j < data.size(); j++) {
            sum += data.at(j) * weights.at(j + 1); // Get the weighted sum of the inputs
        }
        sum += weights.front(); // Add the constant
        neuron_output.at(current_data_id++) = activation(); // Save the calculated values
    }
}

void Perceptron::weight_init() {
    // Check if parameters are valid
    if (input_data.empty() or input_data.at(0).empty()) {
        throw std::invalid_argument("No input data provided.\n"
                                    "Please enter data before initializing the weights.");
    }

    if (weights.size() != input_data.at(0).size() + 1) {
        weights.resize(input_data.at(0).size() + 1);
    }

    // Random number generation
    std::random_device rg; // Create a random device instance
    std::seed_seq seed{rg(), rg(), rg(), rg(), rg(), rg(), rg(), rg()}; // Create a seed
    std::mt19937 eng{seed}; // Get a random seed
    std::uniform_real_distribution<> rand_gen(0, 1); // Generate the random numbers in [0, 1]

    for (std::size_t i = 0; i < input_data.at(0).size() + 1; i++) {
        weights.at(i) = rand_gen(eng); // Assign a random number as the initial weight
    }
}

#if !SIMPLE_ACTIVATION

void Perceptron::update_weights() {
    // Check if parameters are valid
    if (input_data.empty() or weights.empty()) {
        throw std::invalid_argument("No input data provided and/or weights are not initialized.\n"
                                    "Please enter data and/or initialize the weights before attempting an update");
    }

    for (std::size_t i = 0; i < input_data.size(); i++) {
        for (std::size_t j = 0; j < weights.size(); j++) {
#if GRADIENT_DESCENT
#if UNIPOLAR_ACTIVATION
            if (j == 0) {
                weights.at(j) += learning_rate * beta * (tags.at(i) - neuron_output.at(i)) *
                                 neuron_output.at(i) * (1.0 - neuron_output.at(i)); // Update the bias input weight
            } else {
                weights.at(j) += learning_rate * beta * (tags.at(i) - neuron_output.at(i)) *
                                 neuron_output.at(i) * (1.0 - neuron_output.at(i)) *
                                 input_data.at(i).at(j - 1); // Update the rest weights
            }
#else
            if (j == 0) {
                weights.at(j) += learning_rate * beta * (tags.at(i) - neuron_output.at(i)) *
                                 (1.0 - neuron_output.at(i) * neuron_output.at(i)); // Update the bias input weight
            } else {
                weights.at(j) += learning_rate * beta * (tags.at(i) - neuron_output.at(i)) *
                                 (1.0 - neuron_output.at(i) * neuron_output.at(i)) *
                                 input_data.at(i).at(j - 1); // Update the rest weights
            }
#endif
#else
            if (j == 0) {
                weights.at(j) += tags.at(i) * learning_rate; // Update the bias input weight
            } else {
                weights.at(j) += tags.at(i) * input_data.at(i).at(j - 1) * learning_rate; // Update the rest weights
            }
#endif
        }
    }
}

#else
void Perceptron::update_weights(double actual_value, std::size_t data_index) {
    for (std::size_t j = 0; j < weights.size(); j++) {
        if (j == 0) {
            weights.at(j) += actual_value; // Update the bias input weight
        } else {
            weights.at(j) += input_data.at(data_index).at(j - 1) * actual_value; // Update the rest weights
        }
    }
}
#endif

void Perceptron::train() {
    double total_error; // Initialize the variable

    // Check for parameter validity
    if (tags.empty()) {
        throw std::invalid_argument("No data tags provided.\n"
                                    "Training is impossible without data tags.");
    } else if (tags.size() != input_data.size()) {
        throw std::invalid_argument("Tags and input data sizes do not match.\n"
                                    "Please provide a tag for each input element.");
    }

    weight_init(); // Initialize the weights needed for classification

    std::cout << "\n\nTraining session started." << std::endl;
    std::cout << "************************************" << std::endl;

    classify(); // Classify before proceeding

#if SIMPLE_ACTIVATION
    for (std::size_t i = 0; i < tags.size(); i++) {
        if (tags.at(i) != neuron_output.at(i)) {
            update_weights(tags.at(i), i); // Update the weights, since we have a wrong classification
            classify(); // Classify first to get the output values

            i = 0; // Reset the count before continuing
            continue; // Continue the iterations
        }
    }
#else
    while (true) {
        classify(); // Classify first to get the output values
        total_error = training_error(); // Get the total error for the classification
        std::cout << "Total error: " << total_error << std::endl; // TODO to be removed

        if (total_error > acceptable_error) {
            update_weights(); // Perform a weight update to train the neuron

            continue; // Continue the iterations
        } else {
            break; // Get out of the loop, once the error is acceptable
        }
    }
#endif
    std::cout << "****Training finished****";
}

#if !SIMPLE_ACTIVATION

double Perceptron::training_error() {
    double total_error = 0.0;

    for (std::size_t i = 0; i < tags.size(); i++) {
        total_error +=
                0.5 * (tags.at(i) * tags.at(i) - 2 * tags.at(i) * neuron_output.at(i) +
                       neuron_output.at(i) * neuron_output.at(i));
    }
    return total_error;
}

#endif
