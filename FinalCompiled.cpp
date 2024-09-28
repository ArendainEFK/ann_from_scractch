#include <iostream> //  Provides input/output functionality.
#include <vector> // Allows the use of dynamic arrays. Essential for neural network building.
#include <cmath> // Provides mathematical functions like exp, max, etc.
#include <cstdlib> //Provides functions for generating random numbers and converting between numeric types. 
#include <ctime> //Provides functions for manipulating dates and times.
#include <string> //Provides support for strings. (For activation function choice)
#include <fstream> //Provides file input/output functionality.
#include <sstream> //Provides string stream functionality.
#include <algorithm> //Provides various algorithms like max_element.
#include <iomanip>  // Provides functions for manipulating the format of output.
#include <chrono>  //  Provides a high-resolution timer for measuring time durations. (For timing functions)
using namespace std;

// Sigmoid activation function and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// ReLU activation function and its derivative
double relu(double x) {
    return max(0.0, x);
}

double reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

/*The building block of this entire project contains all the attributes that makes
up a neuron including weights, bias, and value. It multiplies each input value by its 
corresponding weight, sums them up with the bias, and applies the chosen activation function 
(either sigmoid or ReLU) to the sum. The result is stored in the value attribute of the neuron.*/
class Neuron {
public:
    vector<double> weights;
    double bias;
    double value;
    Neuron(int num_inputs) {
        for (int i = 0; i < num_inputs; ++i) {
            weights.push_back(((double)rand() / RAND_MAX) * 2 - 1); // Random between -1 and 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    double activate(const vector<double>& inputs, const string& activation_function) {
        double sum = 0.0;
        for (int i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        sum += bias;
        value = (activation_function == "sigmoid") ? sigmoid(sum) : relu(sum);
        return value;
    }
};

// A basic implementation of layering neural to make a neural network.
class Layer {
public:
    vector<Neuron> neurons;
    Layer(int num_neurons, int num_inputs_per_neuron) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(num_inputs_per_neuron);
        }
    }
    vector<double> forward(const vector<double>& inputs, const string& activation_function) {
        vector<double> outputs;
        for (Neuron& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs, activation_function));
        }
        return outputs;
    }
};

// Function to get user input for hyperparameters. Basically catches the input of the user to be used for hyperparameters
void get_hyperparameters(int& num_hidden_neurons, int& num_hidden_layers, double& learning_rate, int& batch_size, int& epochs, string& activation_function) {
    cout << "Enter the number of neurons in the hidden layer: ";
    cin >> num_hidden_neurons;
    cout << "Enter the number of hidden layers: ";
    cin >> num_hidden_layers;
    cout << "Enter the learning rate (e.g., 0.01): ";
    cin >> learning_rate;
    cout << "Enter the batch size: ";
    cin >> batch_size;
    cout << "Enter the number of epochs: ";
    cin >> epochs;
    cout << "Choose activation function (sigmoid/relu): ";
    cin >> activation_function;
}

// Load MNIST dataset
void load_mnist_dataset(vector<vector<double>>& images, vector<int>& labels, const string& image_file, const string& label_file) {
    ifstream image_file_stream(image_file, ios::binary);
    if (!image_file_stream) {
        cerr << "Could not open the image file: " << image_file << endl;
        return;
    }

    image_file_stream.seekg(16);  // Skip header
    unsigned char pixel[784];
    while (image_file_stream.read(reinterpret_cast<char*>(pixel), sizeof(pixel))) {
        vector<double> image;
        for (int i = 0; i < 784; ++i) {
            image.push_back(pixel[i] / 255.0); // Normalize pixel values. Ensure that the dataset is normalized before passing the neural network
        }
        images.push_back(image);
    }

    ifstream label_file_stream(label_file, ios::binary);
    if (!label_file_stream) {
        cerr << "Could not open the label file: " << label_file << endl;
        return;
    }

    label_file_stream.seekg(8);  // Skips dataset header to ensure sparsity within the dataset as it contains the label
    unsigned char label;
    while (label_file_stream.read(reinterpret_cast<char*>(&label), sizeof(label))) {
        labels.push_back(label);
    }
}

// Function that evaluates the model accuracy based on the user input.
double evaluate_model(const vector<vector<double>>& images, const vector<int>& labels, Layer& output_layer, const string& activation_function) {
    int correct_predictions = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        vector<double> final_output = output_layer.forward(images[i], activation_function);
        int predicted_digit = max_element(final_output.begin(), final_output.end()) - final_output.begin();
        if (predicted_digit == labels[i]) {
            correct_predictions++;
        }
    }

    return static_cast<double>(correct_predictions) / labels.size() * 100;
}

/*Function where the user is allowed to test his neural network model by entering an integer between
0-9 (MNIST dataset content). It also contains error handling if the user enters an integer that is
out of scope as well as the option to close and exit the program*/
void test_model(const vector<vector<double>>& images, const vector<int>& labels, Layer& output_layer, const string& activation_function) {
    int selected_digit;
    char choice;

    do {
        // Input validation loop for digit selection
        while (true) {
            cout << "Enter a digit (0-9) to test (or any other key to exit): ";
            if (!(cin >> selected_digit)) {
                // If the input is not an integer, exit the loop
                cout << "Exiting test mode." << endl;
                return;
            }

            // Check if the input is within the valid range (0-9)
            if (selected_digit >= 0 && selected_digit <= 9) {
                break;  // Valid input, exit the loop
            } else {
                cout << "Invalid input! Please enter a valid digit between 0 and 9." << endl;
                cin.clear();  // Clear the error state
                cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Discard invalid input
            }
        }

        bool found = false;
        for (size_t i = 0; i < images.size(); ++i) {
            if (labels[i] == selected_digit) {
                vector<double> final_output = output_layer.forward(images[i], activation_function);
                int predicted_digit = max_element(final_output.begin(), final_output.end()) - final_output.begin();
                cout << "Predicted digit for " << selected_digit << ": " << predicted_digit << endl;
                found = true;
                break;
            }
        }

        if (!found) {
            cout << "Digit " << selected_digit << " not found in the dataset." << endl;
        }

        cout << "Do you want to test another digit? (y/n): ";
        cin >> choice;

    } while (choice == 'y' || choice == 'Y');
    
    cout << "Exiting the test function." << endl;
}
// Main function
int main() {
    srand(static_cast<unsigned int>(time(0)));

    // Get hyperparameters
    int num_hidden_neurons, num_hidden_layers, epochs, batch_size;
    double learning_rate;
    string activation_function;
    get_hyperparameters(num_hidden_neurons, num_hidden_layers, learning_rate, batch_size, epochs, activation_function);

    // Start timing the entire process
    auto program_start_time = chrono::high_resolution_clock::now();

    // Define layers of the network
    Layer input_layer(784, 0);  // Input layer for MNIST (28*28 dimensions of the image)
    /*Input layer has a default value to ensure that the program works properly when interacted by a user.
    This is done due to most of the users lacking knowledge about the MNIST dataset and what it contains.
    Also done as a fail safe so that the user won't have any problems when entering a value for their
    hidden layers*/
    vector<Layer> hidden_layers;
    for (int i = 0; i < num_hidden_layers; ++i) {
        hidden_layers.push_back(Layer(num_hidden_neurons, i == 0 ? 784 : num_hidden_neurons));
    }
    Layer output_layer(10, num_hidden_neurons);  // Output layer for 10 digits

    // Loading the dataset contains the whole MNIST dataset in binary format for faster implementation and better performance
    vector<vector<double>> train_images, test_images;
    vector<int> train_labels, test_labels;
    load_mnist_dataset(train_images, train_labels, "train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    load_mnist_dataset(test_images, test_labels, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        cout << "Epoch " << epoch + 1 << "/" << epochs << " [";
        for (int i = 0; i < 50; ++i) {
            cout << "#";  // Progress bar
        }
        cout << "]" << endl;

        //Lacks backward propagation due to high level complexity I can't figure out how to implement it yet
    }

    // Evaluating the model accuracy printed in percentile value for better understanding
    double train_accuracy = evaluate_model(train_images, train_labels, output_layer, activation_function);
    cout << "Training accuracy: " << fixed << setprecision(2) << train_accuracy << "%" << endl;

    double test_accuracy = evaluate_model(test_images, test_labels, output_layer, activation_function);
    cout << "Testing accuracy: " << fixed << setprecision(2) << test_accuracy << "%" << endl;

     // End timing the entire process
    auto program_end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> total_duration = program_end_time - program_start_time;

    // Output total execution time
    cout << "Total execution time from network generation to training: " 
         << fixed << setprecision(2) << total_duration.count() << " seconds." << endl;

    // Test function is called to enable user testing
    test_model(test_images, test_labels, output_layer, activation_function);

    cout.flush();  // Ensure output is flushed to console
    system("pause");
    cout << endl;
    return 0;
}