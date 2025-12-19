# Federated Learning Project

This repository contains a simple **Federated Learning** simulation implemented in Python.

## Project Overview
Federated Learning allows training a machine learning model using **data distributed across multiple locations** without centralizing the data. In this project, we simulate this process on a single machine:

- **model.py** — defines the model architecture that will be trained.
- **server.py** — coordinates the training process, aggregating updates from clients.
- **client.py** — simulates a client holding a portion of the data and performing local training.

This project uses MNIST dataset to train a simple MLP model ! 

Also in this repository, there is a PDF file that I have reviewed everything about federated learning, you can first read it.

> **Note:** In this simulation, all clients run locally on my machine for demonstration purposes. In a real-world scenario, clients would be distributed across different devices or locations.

## Running the Project
You can run this project using four terminals (or similar environments):

In the terminals, write:
```bash
python server.py
python client.py 0
python client.py 1
python client.py 2
```

finally you can see the results and accuracy of the trained model
