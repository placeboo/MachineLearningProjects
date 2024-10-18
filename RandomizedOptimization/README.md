# Randomized Optimization Algorithms

This project explores the performance of randomized optimization algorithms in two domains: discrete optimization problems and neural network weight optimization. It implements and compares Random Hill Climbing (RHC), Simulated Annealing (SA), Genetic Algorithm (GA), and MIMIC on various problem instances.

## Project Structure

- `main_4peak.py`: Experiments for the Four-Peaks problem
- `main_nqueen.py`: Experiments for the N-Queens problem
- `main_nn.py`: Neural network weight optimization experiments
- `optimization.py`: Implementation of optimization algorithms
- `utils.py`: Utility functions for data processing and visualization
- `playground/`: Jupyter notebooks for data exploration and plot generation
  - `4peak_plots.ipynb`: Notebook for generating Four-Peaks problem plots
  - `nqueue_plots.ipynb`: Notebook for generating N-Queens problem plots
  - `nn_plots.ipynb`: Notebook for generating neural network optimization plots

## Setup

1. Clone this repository.
2. Create a conda environment using the provided `environment.yml` file:

   ```
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```
   conda activate RandomizedOptimization
   ```

4. To use the Jupyter notebooks, ensure you have Jupyter installed in your environment:

   ```
   conda install jupyter
   ```

## Running Experiments

### Discrete Optimization Problems

To run experiments for the Four-Peaks problem:

```
python main_4peak.py
```

To run experiments for the N-Queens problem:

```
python main_nqueen.py
```

### Neural Network Weight Optimization

To run neural network weight optimization experiments:

```
python main_nn.py
```

### Generating Plots

To generate plots and explore the data:

1. Navigate to the `playground` directory:

   ```
   cd playground
   ```

2. Start Jupyter Notebook:

   ```
   jupyter notebook
   ```

3. Open the desired notebook (`4peak_plots.ipynb` or `nqueue_plots.ipynb` or `nn_plots.ipynb`) and run the cells to generate plots.

## Experiments

The project includes the following experiments:

1. Four-Peaks Problem:
   - Performance comparison across problem sizes
   - Hyperparameter tuning for RHC, SA, GA, and MIMIC

2. N-Queens Problem:
   - Performance comparison across problem sizes
   - Hyperparameter tuning for RHC, SA, and GA

3. Neural Network Weight Optimization:
   - Comparison of RHC, SA, and GA against backpropagation
   - Hyperparameter tuning for each algorithm

## Results

Results are saved in the `results` directory, and visualizations are stored in the `figures` directory. These include performance metrics, execution times, and fitness curves for various problem sizes and hyperparameter configurations.

The Jupyter notebooks in the `playground` folder provide interactive visualizations and allow for further exploration of the experimental results.


## Author
Jackie Jiaqi Yin

## Acknowledgments

This project was completed as part of the CS7641 Machine Learning course at Georgia Tech.