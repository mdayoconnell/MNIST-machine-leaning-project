# Created by Micah
# Date: 10/14/25
# Time: 10:59 AM
# Project: NumpyNetwork
# File: __main__.py

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
csv_path = DATA_DIR / "train.csv"

# Dynamically finds the path for the files that initialize our data
DATA_MODULE_PATH = ROOT / "src" / "data" / "mnist_csv.py"
MODEL_MODULES = {
    "mlp_2layer": ROOT / "src" / "models" / "mlp_2layer.py",
}


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_mnist_experiment(csv_path: Path, model_name: str, hidden_exponents, cref_fraction=0.1):
    data_mod = load_module(DATA_MODULE_PATH, "mnist_csv")
    model_mod = load_module(MODEL_MODULES[model_name], model_name.replace("-", "_"))

    X, Y = data_mod.create_mnist_csv(csv_path)
    X_train, Y_train, X_cref, Y_cref = data_mod.split_data(X, Y, cref_fraction=cref_fraction)

    histories = {}
    for hidden_exp in hidden_exponents:
        n_hidden = 2 ** hidden_exp
        print("training with hidden units: ", n_hidden)
        _, _, _, _, history = model_mod.gradient_descent(
            X_train, Y_train, X_cref, Y_cref, n_hidden=n_hidden,
        )
        histories[n_hidden] = history

    return histories


def plot_histories(histories, title):
    for h, hist in histories.items():
        iters = [t for t, _ in hist]
        accs = [a for _, a in hist]
        plt.plot(iters, accs, label=f'hidden neurons: {h}')
    plt.xlabel("Iteration")
    plt.ylabel("Cross-reference accuracy")
    plt.title(title)
    plt.legend()


def main():
    csv_path = Path("/Users/micah/kaggle/digit-recognizer/train.csv")
    histories = run_mnist_experiment(
        csv_path=csv_path,
        model_name="mlp_2layer",
        hidden_exponents=range(6, 7),
        cref_fraction=0.1,
    )
    plot_histories(histories, "MNIST accuracy vs number of forward training propagations")
    plt.show()


if __name__ == "__main__":
    main()
