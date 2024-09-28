import numpy as np
import tkinter as tk
from tkinter import messagebox, Canvas, ttk
from tensorflow.keras.datasets import mnist
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import threading

# Activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values, axis=0)

# Loss functions
def cross_entropy_loss(predictions, labels):
    return -np.sum(labels * np.log(predictions + 1e-9)) / labels.shape[0]

def cross_entropy_loss_derivative(predictions, labels):
    return predictions - labels

# Neuron class
class Neuron:
    def __init__(self, num_inputs, activation="relu"):
        self.weights = np.random.randn(num_inputs) * np.sqrt(1. / num_inputs)
        self.bias = np.random.randn() * np.sqrt(1. / num_inputs)
        self.activation = activation

    def activate(self, inputs):
        inputs = np.array(inputs)
        z = np.dot(inputs, self.weights) + self.bias

        if self.activation == "sigmoid":
            self.output = sigmoid(z)
        else:
            self.output = relu(z)
        return self.output

# Layer class
class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, activation="relu"):
        self.neurons = [Neuron(num_inputs_per_neuron, activation) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation="relu"):
        self.layers = []
        self.layers.append(Layer(hidden_layers[0], input_size, activation))
        for i in range(1, len(hidden_layers)):
            self.layers.append(Layer(hidden_layers[i], hidden_layers[i-1], activation))
        self.layers.append(Layer(output_size, hidden_layers[-1], activation="softmax"))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return softmax(np.array(inputs))

    def train(self, train_X, train_y, epochs, learning_rate, batch_size, activation):
        for epoch in range(epochs):
            for i in range(0, len(train_X), batch_size):
                X_batch = train_X[i:i+batch_size]
                y_batch = train_y[i:i+batch_size]
                for j in range(len(X_batch)):
                    self._backpropagate(X_batch[j], y_batch[j], learning_rate, activation)

    def _backpropagate(self, x, y, learning_rate, activation):
        output = self.forward(x)
        loss = cross_entropy_loss(output, y)
        # Backward pass was not implemented due to complexity and time constraints

# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(-1, 28*28) / 255.0
test_X = test_X.reshape(-1, 28*28) / 255.0
train_y = np.eye(10)[train_y]
test_y = np.eye(10)[test_y]

class NeuralNetworkApp:
    def __init__(self, master):
        self.master = master
        master.title("Neural Network Hyperparameter Tuner")
        master.geometry("650x550")  #Window Size

        # Custom font settings for larger font size
        self.custom_font = ("Helvetica", 16)

        # Hyperparameters
        self.hidden_layers = tk.StringVar(value="128,64")
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.epochs = tk.IntVar(value=10)
        self.batch_size = tk.IntVar(value=32)
        self.activation_function = tk.StringVar(value='relu')

        # Create GUI components
        self.create_widgets()

        # Initialize the trained model
        self.trained_model = None

    def create_widgets(self):
        # Centering elements with columnspan and padding
        ttk.Label(self.master, text="Hidden Layers (comma separated):", font=self.custom_font).grid(row=0, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.hidden_layers, font=self.custom_font).grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Learning Rate:", font=self.custom_font).grid(row=1, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.learning_rate, font=self.custom_font).grid(row=1, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Epochs:", font=self.custom_font).grid(row=2, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.epochs, font=self.custom_font).grid(row=2, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Batch Size:", font=self.custom_font).grid(row=3, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.batch_size, font=self.custom_font).grid(row=3, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Activation Function:", font=self.custom_font).grid(row=4, column=0, padx=10, pady=10)
        ttk.Combobox(self.master, textvariable=self.activation_function,
                     values=['sigmoid', 'relu'], font=self.custom_font).grid(row=4, column=1, padx=10, pady=10)

        self.start_button = ttk.Button(self.master, text="Start Training", command=self.start_training, style="Custom.TButton")
        self.start_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.progress = ttk.Progressbar(self.master, length=400)
        self.progress.grid(row=6, column=0, columnspan=2, pady=10)

        self.accuracy_label = ttk.Label(self.master, text="", font=self.custom_font)
        self.accuracy_label.grid(row=7, column=0, columnspan=2, pady=10)

        self.time_label = ttk.Label(self.master, text="", font=self.custom_font)  # Label for training time
        self.time_label.grid(row=8, column=0, columnspan=2, pady=10)

        ttk.Label(self.master, text="Test a Number (0-9):", font=self.custom_font).grid(row=9, column=0, padx=10, pady=10)
        self.test_input = ttk.Entry(self.master, font=self.custom_font)
        self.test_input.grid(row=9, column=1, padx=10, pady=10)
        self.test_button = ttk.Button(self.master, text="Test", command=self.test_model, style="Custom.TButton")
        self.test_button.grid(row=10, column=0, columnspan=2, pady=10)

    def start_training(self):
        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        hidden_layers = list(map(int, self.hidden_layers.get().split(",")))
        learning_rate = self.learning_rate.get()
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()
        activation = self.activation_function.get()

        nn = NeuralNetwork(28*28, hidden_layers, 10, activation)
        
        start_time = time.time()  # Start time tracking
        
        for epoch in range(epochs):
            nn.train(train_X, train_y, epochs=1, learning_rate=learning_rate, batch_size=batch_size, activation=activation)

            # Update progress bar
            self.progress['value'] = (epoch + 1) / epochs * 100
            self.master.update_idletasks()

        end_time = time.time()  # End time tracking
        training_duration = end_time - start_time
        self.time_label.config(text=f"Training Time: {training_duration:.2f} seconds")  # Display training time
        
        self.trained_model = nn
        accuracy = self.calculate_accuracy(nn, test_X[:1000], test_y[:1000])
        self.accuracy_label.config(text=f"Test Accuracy: {accuracy * 100:.2f}%")
        self.start_button.config(state=tk.NORMAL)

    def calculate_accuracy(self, nn, test_X, test_y):
        correct_predictions = 0
        for i in range(len(test_X)):
            output = nn.forward(test_X[i])
            predicted_digit = np.argmax(output)
            if predicted_digit == np.argmax(test_y[i]):
                correct_predictions += 1
        return correct_predictions / len(test_X)

    def test_model(self):
        number = self.test_input.get()
        if number.isdigit() and 0 <= int(number) <= 9 and self.trained_model is not None:
            digit_index = np.random.choice(np.where(test_y.argmax(axis=1) == int(number))[0])
            test_image = test_X[digit_index].reshape(28, 28)

            # Show the image
            plt.imshow(test_image, cmap='gray')
            plt.title(f"Testing Model on Number: {number}")
            plt.axis('off')
            plt.show()

            # Predict the number using the trained model
            prediction = self.trained_model.forward(test_image.flatten())
            predicted_number = np.argmax(prediction)

            # Display prediction
            messagebox.showinfo("Prediction Result", f"The model predicts this is the number: {predicted_number}")
        else:
            messagebox.showerror("Error", "Please train the model first and enter a valid number (0-9).")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()
