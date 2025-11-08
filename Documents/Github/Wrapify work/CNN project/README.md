# fashion-mnist-playground

Hands-on experiments with Fashion-MNIST using PyTorch. The project lives primarily in the Jupyter notebook `fashion_mnist_cnn.ipynb`, which walks through data exploration, model training, and evaluation.

## Project Highlights
- Loads Fashion-MNIST with normalization and quick visual checks.
- Trains a baseline multilayer perceptron (MLP) and records accuracy trends.
- Builds a convolutional neural network (CNN) that improves performance, saving the best weights.
- Generates confusion matrices and misclassified-image grids for error analysis.
- Summarizes results and outlines recommended next steps.

## Getting Started
```bash
# clone the repo
git clone https://github.com/SMariamD/Fashion-Mnist-Playground.git
cd Fashion-Mnist-Playground

# create and activate a virtual environment (PowerShell example)
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Running the Experiments
- Launch Jupyter or VS Code and open `fashion_mnist_cnn.ipynb`.
- Execute cells sequentially to download the dataset, train models, visualize metrics, and inspect artifacts.
- Saved outputs (best checkpoints, confusion matrices) are written to the `artifacts/` directory.

## Repository Structure
- `fashion_mnist_cnn.ipynb` — Primary notebook containing data prep, models, training loops, and analysis.
- `requirements.txt` — Pinned Python packages for reproducibility.
- `artifacts/` — Created at runtime to store model weights and plots.

## Next Steps
- Introduce a dedicated validation split for hyperparameter tuning.
- Explore regularization techniques such as data augmentation, dropout adjustments, or batch normalization tweaks.
- Iterate on architectures or learning rate schedules once the baseline is solid.

Contributions and extensions are welcome—feel free to fork the project or open an issue to suggest enhancements.

