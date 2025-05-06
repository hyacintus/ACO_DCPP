# ACO-DCPP
# Project Overview
This project consists of three primary Python scripts designed to work together in solving a graph generation and optimization problem. The objective is to generate a directed graph with a given number of nodes, including a desired number of odd nodes, and a specified maximum number of edges. The problem is addressed using both a recursive algorithm to calculate all possible solutions and an Ant Colony Optimization (ACO) algorithm to optimize the best solution. The project provides a useful framework for graph-related computations and algorithm testing.

# File Description
# 1. loop-p.py
This script is responsible for running the main script, Loop_per_articolo.py, multiple times in an automated manner. It is particularly useful for conducting repeated tests or analyses without manually initiating the script each time.

# 2. Loop_per_articolo.py
This is the main script that generates a directed graph with a specified number of nodes, a desired number of odd nodes, and a maximum number of edges. The script solves the problem using two methods:
- A recursive algorithm, which calculates all possible solutions to the problem.
- An Ant Colony Optimization (ACO) algorithm, which aims to optimize the solution found by the recursive algorithm.

# 3. funzioni_per_articolo.py
This file contains all the functions used within the main script, Loop_per_articolo.py. These functions are modular and handle various tasks, such as graph generation, recursive solution computation, and the implementation of the Ant Colony Optimization algorithm.

# Additional Details
The files in this project are fully commented in Italian. The comments aim to make the code easier to understand, particularly for anyone who may want to modify or extend the functionality. Additionally, some sections of the code are commented out. These parts are not necessary for the current functioning of the scripts but may serve as useful references or starting points for future modifications. Users can adapt these sections to customize the scripts to their needs.
