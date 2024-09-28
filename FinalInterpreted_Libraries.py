from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time  # Import time module

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class NeuralNetworkApp:
    def __init__(self, master):
        self.master = master
        master.title("Neural Network Hyperparameter Tuner")
        master.geometry("500x600")  # Adjust window size

        # Font size and style
        self.custom_font = ("Helvetica", 16)

        # Hyperparameters
        self.hidden_units = tk.IntVar(value=64)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.epochs = tk.IntVar(value=20)
        self.batch_size = tk.IntVar(value=64)
        self.dropout_rate = tk.DoubleVar(value=0.2)
        
        # Activation function
        self.activation_function = tk.StringVar(value='relu')

        # Create GUI components
        self.create_widgets()

        # Initialize the trained model
        self.trained_model = None

    def create_widgets(self):
        ttk.Label(self.master, text="Hidden Units:", font=self.custom_font).grid(row=0, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.hidden_units, font=self.custom_font).grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Learning Rate:", font=self.custom_font).grid(row=1, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.learning_rate, font=self.custom_font).grid(row=1, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Epochs:", font=self.custom_font).grid(row=2, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.epochs, font=self.custom_font).grid(row=2, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Batch Size:", font=self.custom_font).grid(row=3, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.batch_size, font=self.custom_font).grid(row=3, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Dropout Rate:", font=self.custom_font).grid(row=4, column=0, padx=10, pady=10)
        ttk.Entry(self.master, textvariable=self.dropout_rate, font=self.custom_font).grid(row=4, column=1, padx=10, pady=10)

        ttk.Label(self.master, text="Activation Function:", font=self.custom_font).grid(row=5, column=0, padx=10, pady=10)
        ttk.Combobox(self.master, textvariable=self.activation_function,
                     values=['sigmoid', 'relu', 'tanh'], font=self.custom_font).grid(row=5, column=1, padx=10, pady=10)

        self.start_button = ttk.Button(self.master, text="Start Training", command=self.start_training, style="Custom.TButton")
        self.start_button.grid(row=6, column=0, columnspan=2, pady=10)

        self.progress = ttk.Progressbar(self.master, length=400)
        self.progress.grid(row=7, column=0, columnspan=2, pady=10)

        self.accuracy_label = ttk.Label(self.master, text="", font=self.custom_font)
        self.accuracy_label.grid(row=8, column=0, columnspan=2, pady=10)

        ttk.Label(self.master, text="Test a Number (0-9):", font=self.custom_font).grid(row=9, column=0, padx=10, pady=10)
        self.test_input = ttk.Entry(self.master, font=self.custom_font)
        self.test_input.grid(row=9, column=1, padx=10, pady=10)
        self.test_button = ttk.Button(self.master, text="Test", command=self.test_model, style="Custom.TButton")
        self.test_button.grid(row=10, column=0, columnspan=2, pady=10)

        self.time_label = ttk.Label(self.master, text="", font=self.custom_font)  # New label to display time
        self.time_label.grid(row=11, column=0, columnspan=2, pady=10)

    def start_training(self):
        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        # Start the timer
        start_time = time.time()

        # Build model
        model = Sequential()
        model.add(Dense(self.hidden_units.get(), input_shape=(28 * 28,), activation=self.activation_function.get()))
        if self.dropout_rate.get() > 0:
            model.add(Dropout(self.dropout_rate.get()))
        model.add(Dense(self.hidden_units.get(), activation=self.activation_function.get()))
        if self.dropout_rate.get() > 0:
            model.add(Dropout(self.dropout_rate.get()))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate.get()),
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(x_train, y_train, epochs=self.epochs.get(), batch_size=self.batch_size.get(),
                  validation_data=(x_test, y_test), 
                  verbose=0, 
                  callbacks=[self.progress_callback(), early_stopping])

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Calculate time taken
        time_taken = time.time() - start_time

        # Update accuracy label as a percentage
        self.accuracy_label.config(text=f"Test Accuracy: {accuracy * 100:.2f}%")

        # Update time label with training time
        self.time_label.config(text=f"Training Time: {time_taken:.2f} seconds")

        # Store the trained model for testing
        self.trained_model = model

        self.start_button.config(state=tk.NORMAL)

    def test_model(self):
        number = self.test_input.get()
        if number.isdigit() and 0 <= int(number) <= 9 and self.trained_model is not None:
            image_index = np.random.choice(np.where(y_test.argmax(axis=1) == int(number))[0])
            image = x_test[image_index].reshape(28, 28)

            # Show the image
            plt.imshow(image, cmap='gray')
            plt.title(f"Testing Model on Number: {number}")
            plt.axis('off')
            plt.show()

            # Predict the number using the trained model
            prediction = self.trained_model.predict(image.reshape(1, 28 * 28))
            predicted_number = np.argmax(prediction)

            # Display prediction
            messagebox.showinfo("Prediction Result", f"The model predicts this is the number: {predicted_number}")
        else:
            messagebox.showerror("Error", "Please train the model first and enter a valid number (0-9).")

    def progress_callback(self):
        class ProgressBar(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
            
            def on_epoch_end(self, epoch, logs=None):
                self.progress_bar['value'] = (epoch + 1) / self.params['epochs'] * 100
                self.progress_bar.update()

        return ProgressBar(self.progress)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()