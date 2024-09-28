# ANNs From Scratch 
## A way to learn how Artificial Neural Networks are implemented in C++, and Python as well as how hyperparameters work within ANNs and how to modify them.
### A Midterms output for CCS 238: Programming Language Subject.

### Description Of The Problem
As a computer science student that majors in artificial intelligence. One of the most tackled topic 
within our course is artificial neural networks. Artificial Neural Networks (ANNs) is an essential 
sub-topic within our course as it serves as a foundation for many machine learning and AI tasks. 
However, due to many constraints within the education system ANNs tend to be hastily thought resulting 
to lesser knowledge about it. Especially when it comes to the different hyperparameters that make 
up ANNs and how to efficiently use and structure them.

### Project Overview
This project aims to build artificial neural networks (ANNs) from scratch in both C++ and Python, 
focusing on modifying and experimenting with hyperparameters such as learning rate, batch size, and 
number of layers. By creating a customizable framework in both languages, users can delve into the 
mechanics of ANNs and observe how changes in hyperparameters impact performance, enabling a deeper 
understanding of their ultimate effects on model accuracy and training efficiency.

### About The Codes
#### C++
This code implements a neural network framework for handwritten digit classification using the MNIST 
dataset. It defines the structure of neurons and layers, allows users to input hyperparameters, 
loads the dataset, and provides functions for evaluating and testing the model. However, the code 
lacks the essential backward propagation algorithm, which is necessary for training the network to 
learn and improve its accuracy. As a result, the current implementation can only be used for testing and 
demonstration purposes.

WIP is the backward propagation algorithm. Did not have the time to study if futher and fully implement
it in C++.

#### Python from Scratch
Basically the same functionality with the C++ code as however the output is different as it makes use
of the tkinter GUI platform built in Python. Also lacks the essential backward propagation algorithm. 

#### Python with Libraries
This code implements a user-friendly application for training a neural network on the MNIST handwritten digit 
dataset. It allows you to define hyperparameters like hidden units, learning rate, and epochs through a 
graphical interface.  After training, you can test the model's performance by inputting a number (0-9) 
and the application will display a random image of that digit and predict the number using the trained model.

This code performs the best among the three as it makes use of the Tensorflow:Keras Libraries. It is 
very efficient, however it has the highest level of abstraction. Good for parameter tuning, and model building.
