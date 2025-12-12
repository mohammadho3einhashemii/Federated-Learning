# Federated Learning Project

This repository contains a simple implementation of **Federated Learning**, a machine learning approach where the model is trained across multiple devices (clients) without centralizing the data. This allows data privacy and distributed computation.

## Project Overview

In Federated Learning, each client has its own local dataset. The server coordinates the training process by aggregating updates from the clients without accessing their raw data. 

This project includes:

- **model.py** — defines the model to be trained.
- **server.py** — coordinates the training process and aggregates updates from clients.
- **client.py** — simulates a client holding a part of the data.
- **Federated_Review.pdf** — a PDF with a comprehensive theoretical overview of Federated Learning.

## How to Run

You will need **four terminals** (or any similar environment):

1. In the first terminal, run the server:
   ```bash
   python server.py
