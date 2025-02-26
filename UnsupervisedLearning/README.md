# Unsupervised Learning and Dimensionality Reduction

This project implements and analyzes various unsupervised learning algorithms and dimensionality reduction techniques.

## Project Structure

```
src/
├── clustering/               # Clustering algorithm implementations
│   ├── kmeans.py           # K-means clustering
│   └── em.py               # Expectation Maximization (GMM)
│
├── dimensionality_reduction/ # Dimensionality reduction implementations
│   ├── pca.py              # Principal Component Analysis
│   ├── ica.py              # Independent Component Analysis
│   ├── random_projection.py # Random Projection
│   └── umap.py             # UMAP algorithm
│
├── experiments/             # Experimental implementations
│   ├── experiment1_clustering.py        # Clustering analysis
│   ├── experiment2_dimensionality.py    # Dimensionality reduction analysis
│   ├── experiment3_combined.py          # Combined clustering and DR
│   ├── experiment4_RdNN.py             # Neural Networks with DR
│   └── experiment5_NN.py               # Neural Network experiments
│
├── neural_network/          # Neural network implementations
│   └── nn_model.py         # Base neural network model
│
└── utils/                   # Utility functions
    ├── data_loader.py      # Data loading utilities
    ├── evaluation.py       # Evaluation metrics
    └── plotting.py         # Visualization utilities
```

## Module Descriptions

### Clustering (`src/clustering/`)
- `kmeans.py`: Implementation of K-means clustering algorithm
- `em.py`: Implementation of Expectation Maximization for Gaussian Mixture Models

### Dimensionality Reduction (`src/dimensionality_reduction/`)
- `pca.py`: Principal Component Analysis implementation
- `ica.py`: Independent Component Analysis implementation
- `random_projection.py`: Random Projection methods
- `umap.py`: Uniform Manifold Approximation and Projection

### Experiments (`src/experiments/`)
1. `experiment1_clustering.py`: 
   - Implements clustering analysis
   - Evaluates K-means and EM algorithms
   - Includes metrics like silhouette score, inertia

2. `experiment2_dimensionality.py`:
   - Analyzes different dimensionality reduction techniques
   - Compares PCA, ICA, Random Projection, and UMAP
   - Evaluates reconstruction error and explained variance

3. `experiment3_combined.py`:
   - Combines clustering with dimensionality reduction
   - Analyzes the impact of DR on clustering performance

4. `experiment4_RdNN.py`:
   - Neural network experiments with dimensionality reduction
   - Studies the effect of DR on neural network performance

5. `experiment5_NN.py`:
   - Additional neural network experiments
   - Focuses on clustering-based neural network analysis

### Neural Network (`src/neural_network/`)
- `nn_model.py`: Base neural network implementation with customizable architectures

### Utilities (`src/utils/`)
- `data_loader.py`: Functions for loading and preprocessing data
- `evaluation.py`: Implementation of evaluation metrics and analysis tools
- `plotting.py`: Visualization functions for results and analysis

## Notebooks

The project includes several Jupyter notebooks for running experiments:

1. `experiment1_clustering.ipynb`: 
   - Demonstrates clustering analysis
   - Visualizes cluster formations
   - Analyzes clustering metrics

2. `experiment2_dimensionality.ipynb`:
   - Shows dimensionality reduction analysis
   - Compares different DR techniques
   - Visualizes reduced dimensions

3. `experiment3_combined.ipynb`:
   - Combines clustering with DR techniques
   - Analyzes the combined performance

4. `experiment4_RdNN.ipynb`:
   - Neural network experiments with DR
   - Performance analysis

5. `experiment5_NN.ipynb`:
   - Advanced neural network experiments
   - Clustering-based neural network analysis

## Usage

1. Install required dependencies:
```bash
conda env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate unsupervised
```

3. Run experiments using the Jupyter notebooks in the notebooks directory:
```bash
jupyter notebook
```

## Results

Results are saved in the `results` directory, and visualizations are stored in the `figs` directory. These include performance metrics, execution times, and fitness curves for various problem sizes and hyperparameter configurations.


## Notes
- All experiments are designed to be reproducible with fixed random seeds
- Visualization functions support both 2D and 3D plotting
- Results are automatically saved in designated directories

## Author
Jackie Jiaqi Yin
