# Federated Learning Project

This repository demonstrates a simple yet illustrative implementation of **Federated Learning (FL)** — a distributed training paradigm where multiple clients collaboratively train a model **without sharing their local data**.

---

## Overview

Federated Learning enables training on data that is geographically distributed or privacy-sensitive.  
Instead of sending raw data to a central server, each client trains locally and only **model updates** are sent to the server. The server then aggregates these updates and forms the global model.

This project includes three main Python components:

- **model.py** – Defines the machine learning model used across all clients.
- **server.py** – Coordinates the training process, aggregates client updates, and maintains the global model.
- **client.py** – Represents each participant in the federated system, performing local training on its own data.

A PDF file is also included, providing a concise theoretical explanation of Federated Learning concepts.

---

## How to Run the Project

The system is designed to simulate one server and multiple clients.  
You can run the full setup using **four terminals** (or CMD/PowerShell windows):

1. **Start the server**
python server.py
 
2. **Start Three clients**
python client.py 0 ___
python client.py 1___
python client.py 2


finally you can see the result and accuracy of trianed model in the server window !