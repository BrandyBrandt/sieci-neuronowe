from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
BASE_DIR = Path(r"C:\Users\olobr\OneDrive\Pulpit\mio\mio6")
DATA_CACHE = {}
RUN_CACHE = {}
CLASS_COLORS = ['#355C7D', '#C06C84', '#6C5B7B', '#F67280', '#F8B195']

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 180)
np.set_printoptions(suppress=True, precision=4)


def stable_seed(*parts) -> int:
    text = '::'.join(map(str, parts))
    value = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)
    return value % (2**32 - 1)


def make_rng(*parts) -> np.random.Generator:
    seed = stable_seed(SEED, *parts)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def read_csv_fixed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    first = str(frame.columns[0]).lower()
    if first.startswith('unnamed') or first == '':
        frame = frame.iloc[:, 1:].copy()
    return frame


def standardize_train_test(x_train: np.ndarray, x_other: np.ndarray):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (x_train - mean) / std, (x_other - mean) / std, mean, std


def standardize_target(y_train: np.ndarray, y_other: np.ndarray):
    mean = y_train.mean(axis=0, keepdims=True)
    std = y_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (y_train - mean) / std, (y_other - mean) / std, mean, std


def one_hot_encode(labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(class_names)}
    encoded = np.zeros((len(labels), len(class_names)), dtype=np.float64)
    for row_idx, label in enumerate(labels.astype(str)):
        encoded[row_idx, index[label]] = 1.0
    return encoded


def train_val_split_regression(x: np.ndarray, y: np.ndarray, val_ratio: float, seed_parts: tuple):
    rng = make_rng('val_reg', *seed_parts)
    order = rng.permutation(len(x))
    val_size = max(1, int(round(len(x) * val_ratio)))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    if len(train_idx) == 0:
        train_idx = order[val_size - 1:]
        val_idx = order[:val_size - 1]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train_val_split_classification(x: np.ndarray, y_labels: np.ndarray, val_ratio: float, seed_parts: tuple):
    rng = make_rng('val_clf', *seed_parts)
    train_idx = []
    val_idx = []
    for label in np.unique(y_labels):
        cls_idx = np.flatnonzero(y_labels == label)
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        cls_val = max(1, int(round(len(cls_idx) * val_ratio)))
        if cls_val >= len(cls_idx):
            cls_val = max(1, len(cls_idx) - 1)
        val_idx.extend(cls_idx[:cls_val].tolist())
        train_idx.extend(cls_idx[cls_val:].tolist())
    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    train_idx = train_idx[np.argsort(train_idx)]
    val_idx = val_idx[np.argsort(val_idx)]
    return x[train_idx], y_labels[train_idx], x[val_idx], y_labels[val_idx]


def load_regression_dataset(name: str, val_ratio: float = 0.2) -> dict:
    train_frame = read_csv_fixed(BASE_DIR / f'{name}-training.csv')
    test_frame = read_csv_fixed(BASE_DIR / f'{name}-test.csv')
    x_train_full = train_frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_train_full = train_frame.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)
    x_test_raw = test_frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_test_raw = test_frame.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)
    x_train_raw, y_train_raw, x_val_raw, y_val_raw = train_val_split_regression(x_train_full, y_train_full, val_ratio, (name, val_ratio))
    x_train, x_val, x_mean, x_std = standardize_train_test(x_train_raw, x_val_raw)
    _, x_test, _, _ = standardize_train_test(x_train_raw, x_test_raw)
    y_train, y_val, y_mean, y_std = standardize_target(y_train_raw, y_val_raw)
    _, y_test, _, _ = standardize_target(y_train_raw, y_test_raw)
    return {
        'name': name,
        'task': 'regression',
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'x_train_raw': x_train_raw,
        'x_val_raw': x_val_raw,
        'x_test_raw': x_test_raw,
        'y_train_raw': y_train_raw,
        'y_val_raw': y_val_raw,
        'y_test_raw': y_test_raw,
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }


def load_classification_dataset(name: str, val_ratio: float = 0.2) -> dict:
    train_frame = read_csv_fixed(BASE_DIR / f'{name}-training.csv')
    test_frame = read_csv_fixed(BASE_DIR / f'{name}-test.csv')
    x_train_full = train_frame[['x', 'y']].to_numpy(dtype=np.float64)
    y_train_full = train_frame['c'].astype(str).to_numpy()
    x_test_raw = test_frame[['x', 'y']].to_numpy(dtype=np.float64)
    y_test_labels = test_frame['c'].astype(str).to_numpy()
    x_train_raw, y_train_labels, x_val_raw, y_val_labels = train_val_split_classification(x_train_full, y_train_full, val_ratio, (name, val_ratio))
    x_train, x_val, x_mean, x_std = standardize_train_test(x_train_raw, x_val_raw)
    _, x_test, _, _ = standardize_train_test(x_train_raw, x_test_raw)
    class_names = sorted(np.unique(np.concatenate([y_train_labels, y_val_labels, y_test_labels])).tolist())
    return {
        'name': name,
        'task': 'classification',
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y_train': one_hot_encode(y_train_labels, class_names),
        'y_val': one_hot_encode(y_val_labels, class_names),
        'y_test': one_hot_encode(y_test_labels, class_names),
        'x_train_raw': x_train_raw,
        'x_val_raw': x_val_raw,
        'x_test_raw': x_test_raw,
        'y_train_labels': y_train_labels,
        'y_val_labels': y_val_labels,
        'y_test_labels': y_test_labels,
        'class_names': class_names,
        'x_mean': x_mean,
        'x_std': x_std,
    }


def get_dataset(name: str, task: str, val_ratio: float = 0.2) -> dict:
    key = (name, task, float(val_ratio))
    if key not in DATA_CACHE:
        if task == 'regression':
            DATA_CACHE[key] = load_regression_dataset(name, val_ratio=val_ratio)
        else:
            DATA_CACHE[key] = load_classification_dataset(name, val_ratio=val_ratio)
    return DATA_CACHE[key]


def preview_dataset(name: str, task: str, val_ratio: float = 0.2) -> pd.DataFrame:
    data = get_dataset(name, task, val_ratio=val_ratio)
    if task == 'regression':
        return pd.DataFrame({'x_train': data['x_train_raw'].ravel(), 'y_train': data['y_train_raw'].ravel()}).head()
    frame = pd.DataFrame(data['x_train_raw'], columns=['x', 'y'])
    frame['c'] = data['y_train_labels']
    return frame.head()

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def activation_forward(name: str, z: np.ndarray) -> np.ndarray:
    if name == 'sigmoid':
        return sigmoid(z)
    if name == 'tanh':
        return np.tanh(z)
    if name == 'relu':
        return np.maximum(0.0, z)
    if name == 'softmax':
        return softmax(z)
    if name == 'linear':
        return z
    raise ValueError(name)


def activation_backward(name: str, z: np.ndarray, activated: np.ndarray) -> np.ndarray:
    if name == 'sigmoid':
        return activated * (1.0 - activated)
    if name == 'tanh':
        return 1.0 - activated**2
    if name == 'relu':
        return (z > 0.0).astype(np.float64)
    if name in {'linear', 'softmax'}:
        return np.ones_like(activated)
    raise ValueError(name)


def init_weights(input_dim: int, output_dim: int, activation: str, rng: np.random.Generator) -> np.ndarray:
    if activation == 'relu':
        return rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, output_dim))
    limit = np.sqrt(6.0 / (input_dim + output_dim))
    return rng.uniform(-limit, limit, size=(input_dim, output_dim))


class DenseLayer:
    def __init__(self, input_dim: int, output_dim: int, activation: str, rng: np.random.Generator, clip_value: float | None = 5.0):
        self.activation = activation
        self.clip_value = clip_value
        self.weights = init_weights(input_dim, output_dim, activation, rng)
        self.biases = np.zeros((1, output_dim), dtype=np.float64)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = inputs @ self.weights + self.biases
        self.output = activation_forward(self.activation, self.z)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.activation not in {'linear', 'softmax'}:
            grad = grad * activation_backward(self.activation, self.z, self.output)
        batch_size = max(1, len(self.inputs))
        self.dweights = self.inputs.T @ grad / batch_size
        self.dbiases = np.sum(grad, axis=0, keepdims=True) / batch_size
        if self.clip_value is not None:
            self.dweights = np.clip(self.dweights, -self.clip_value, self.clip_value)
            self.dbiases = np.clip(self.dbiases, -self.clip_value, self.clip_value)
        return grad @ self.weights.T


class RMSProp:
    def __init__(self, lr: float = 0.001, decay: float = 0.92, epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.c_w = {}
        self.c_b = {}

    def update(self, layers: list[DenseLayer]) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.c_w:
                self.c_w[idx] = np.zeros_like(layer.weights)
                self.c_b[idx] = np.zeros_like(layer.biases)
            grad_w = layer.dweights + self.weight_decay * layer.weights
            grad_b = layer.dbiases
            self.c_w[idx] = self.decay * self.c_w[idx] + (1.0 - self.decay) * (grad_w**2)
            self.c_b[idx] = self.decay * self.c_b[idx] + (1.0 - self.decay) * (grad_b**2)
            layer.weights -= self.lr * grad_w / (np.sqrt(self.c_w[idx]) + self.epsilon)
            layer.biases -= self.lr * grad_b / (np.sqrt(self.c_b[idx]) + self.epsilon)


class Momentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v_w = {}
        self.v_b = {}

    def update(self, layers: list[DenseLayer]) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.v_w:
                self.v_w[idx] = np.zeros_like(layer.weights)
                self.v_b[idx] = np.zeros_like(layer.biases)

            grad_w = layer.dweights + self.weight_decay * layer.weights
            grad_b = layer.dbiases

            self.v_w[idx] = self.momentum * self.v_w[idx] - self.lr * grad_w
            self.v_b[idx] = self.momentum * self.v_b[idx] - self.lr * grad_b

            layer.weights += self.v_w[idx]
            layer.biases += self.v_b[idx]


class MLP:
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, hidden_activation: str, output_activation: str, seed_parts: tuple, clip_value: float | None = 5.0):
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        rng = make_rng('mlp', *seed_parts)
        self.layers = []
        for idx in range(len(dims) - 1):
            activation = output_activation if idx == len(dims) - 2 else hidden_activation
            self.layers.append(DenseLayer(dims[idx], dims[idx + 1], activation, rng, clip_value=clip_value))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, grad: np.ndarray) -> None:
        current = grad
        for layer in reversed(self.layers):
            current = layer.backward(current)

    def get_state(self):
        return [(layer.weights.copy(), layer.biases.copy()) for layer in self.layers]

    def set_state(self, state) -> None:
        for layer, (weights, biases) in zip(self.layers, state):
            layer.weights = weights.copy()
            layer.biases = biases.copy()


def mse_raw(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def cross_entropy(targets: np.ndarray, predictions: np.ndarray) -> float:
    clipped = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(np.sum(targets * np.log(clipped), axis=1)))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> float:
    scores = []
    for class_name in class_names:
        tp = np.sum((y_true == class_name) & (y_pred == class_name))
        fp = np.sum((y_true != class_name) & (y_pred == class_name))
        fn = np.sum((y_true == class_name) & (y_pred != class_name))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        scores.append(2.0 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(scores))


def confusion_matrix_array(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(class_names)}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[index[true_label], index[pred_label]] += 1
    return matrix


def hidden_label(hidden_dims: list[int]) -> str:
    return '-'.join(map(str, hidden_dims))


def count_params(input_dim: int, hidden_dims: list[int], output_dim: int) -> int:
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    return int(sum(left * right + right for left, right in zip(dims[:-1], dims[1:])))

def evaluate_regression(model: MLP, data: dict) -> dict:
    train_pred_scaled = model.forward(data['x_train'])
    val_pred_scaled = model.forward(data['x_val'])
    test_pred_scaled = model.forward(data['x_test'])
    train_loss = float(np.mean((train_pred_scaled - data['y_train']) ** 2))
    val_loss = float(np.mean((val_pred_scaled - data['y_val']) ** 2))
    test_loss_scaled = float(np.mean((test_pred_scaled - data['y_test']) ** 2))
    val_pred_raw = val_pred_scaled * data['y_std'] + data['y_mean']
    test_pred_raw = test_pred_scaled * data['y_std'] + data['y_mean']
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss_scaled': test_loss_scaled,
        'val_mse_raw': mse_raw(data['y_val_raw'], val_pred_raw),
        'test_mse_raw': mse_raw(data['y_test_raw'], test_pred_raw),
        'val_pred_raw': val_pred_raw,
        'test_pred_raw': test_pred_raw,
    }


def evaluate_classification(model: MLP, data: dict) -> dict:
    train_pred = model.forward(data['x_train'])
    val_pred = model.forward(data['x_val'])
    test_pred = model.forward(data['x_test'])
    train_loss = cross_entropy(data['y_train'], train_pred)
    val_loss = cross_entropy(data['y_val'], val_pred)
    test_loss = cross_entropy(data['y_test'], test_pred)
    val_labels = np.array([data['class_names'][idx] for idx in np.argmax(val_pred, axis=1)])
    test_labels = np.array([data['class_names'][idx] for idx in np.argmax(test_pred, axis=1)])
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'val_f1': macro_f1_score(data['y_val_labels'], val_labels, data['class_names']),
        'test_f1': macro_f1_score(data['y_test_labels'], test_labels, data['class_names']),
        'test_labels': test_labels,
        'confusion_matrix': confusion_matrix_array(data['y_test_labels'], test_labels, data['class_names']),
    }


def variant_config(name: str, weight_decay: float = 0.0, early_stopping: bool = False, patience: int = 120):
    return {
        'variant': name,
        'weight_decay': weight_decay,
        'early_stopping': early_stopping,
        'patience': patience,
    }


def build_optimizer(name: str, lr: float, decay: float, momentum: float, weight_decay: float):
    if name == 'rmsprop':
        return RMSProp(lr=lr, decay=decay, weight_decay=weight_decay)
    if name == 'momentum':
        return Momentum(lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def make_run_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def fit_model(dataset_name: str, task: str, hidden_dims: list[int], hidden_activation: str, epochs: int, batch_size: int, lr: float, weight_decay: float = 0.0, early_stopping: bool = False, patience: int = 120, decay: float = 0.92, momentum: float = 0.9, optimizer_name: str = 'rmsprop', clip_value: float | None = 5.0, val_ratio: float = 0.2, eval_every: int = 20, variant: str | None = None, force: bool = False) -> dict:
    spec = {
        'dataset_name': dataset_name,
        'task': task,
        'hidden_dims': list(hidden_dims),
        'hidden_activation': hidden_activation,
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'lr': float(lr),
        'weight_decay': float(weight_decay),
        'early_stopping': bool(early_stopping),
        'patience': int(patience),
        'decay': float(decay),
        'momentum': float(momentum),
        'optimizer_name': optimizer_name,
        'clip_value': None if clip_value is None else float(clip_value),
        'val_ratio': float(val_ratio),
        'eval_every': int(eval_every),
    }
    if variant is not None:
        spec['variant'] = variant
    key = make_run_key(spec)
    if key in RUN_CACHE and not force:
        return copy.deepcopy(RUN_CACHE[key])

    data = get_dataset(dataset_name, task, val_ratio=val_ratio)
    input_dim = data['x_train'].shape[1]
    output_dim = 1 if task == 'regression' else len(data['class_names'])
    output_activation = 'linear' if task == 'regression' else 'softmax'
    params = count_params(input_dim, hidden_dims, output_dim)
    model = MLP(input_dim, list(hidden_dims), output_dim, hidden_activation, output_activation, (dataset_name, task, tuple(hidden_dims), hidden_activation, epochs, batch_size, lr, weight_decay, early_stopping), clip_value=clip_value)
    optimizer = build_optimizer(optimizer_name, lr=lr, decay=decay, momentum=momentum, weight_decay=weight_decay)
    rng = make_rng('train', dataset_name, task, tuple(hidden_dims), hidden_activation, epochs, batch_size, lr, weight_decay, early_stopping)

    x_train = data['x_train']
    y_train = data['y_train']
    n_samples = len(x_train)
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'test_loss': [], 'score': []}
    if task == 'regression':
        best_score = np.inf
    else:
        best_score = -np.inf
    best_state = model.get_state()
    best_snapshot = None
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_samples)
        x_shuffled = x_train[order]
        y_shuffled = y_train[order]
        for start in range(0, n_samples, batch_size):
            stop = start + batch_size
            xb = x_shuffled[start:stop]
            yb = y_shuffled[start:stop]
            predictions = model.forward(xb)
            grad = 2.0 * (predictions - yb) if task == 'regression' else predictions - yb
            model.backward(grad)
            optimizer.update(model.layers)

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            snapshot = evaluate_regression(model, data) if task == 'regression' else evaluate_classification(model, data)
            if task == 'regression':
                score = snapshot['val_mse_raw']
                test_loss = snapshot['test_loss_scaled']
                better = score < best_score - 1e-10
            else:
                score = snapshot['val_f1']
                test_loss = snapshot['test_loss']
                better = score > best_score + 1e-10
            history['epoch'].append(epoch)
            history['train_loss'].append(snapshot['train_loss'])
            history['val_loss'].append(snapshot['val_loss'])
            history['test_loss'].append(test_loss)
            history['score'].append(score)
            if better:
                best_score = score
                best_state = model.get_state()
                best_snapshot = copy.deepcopy(snapshot)
                best_epoch = epoch
            if early_stopping and epoch - best_epoch >= patience:
                break

    model.set_state(best_state)
    result = {
        'config': spec,
        'dataset_name': dataset_name,
        'task': task,
        'hidden_label': hidden_label(hidden_dims),
        'depth': len(hidden_dims),
        'params': params,
        'seconds': time.perf_counter() - started,
        'stopped_epoch': history['epoch'][-1],
        'best_epoch': best_epoch,
        'history': history,
        'best': best_snapshot,
        'model': model,
    }
    RUN_CACHE[key] = copy.deepcopy(result)
    return copy.deepcopy(result)


def run_variant_suite(dataset_name: str, task: str, model_config: dict, variants: list[dict], force: bool = False) -> list[dict]:
    results = []
    for variant in variants:
        cfg = dict(model_config)
        cfg.update(variant)
        results.append(fit_model(dataset_name=dataset_name, task=task, force=force, **cfg))
    return results


def run_architecture_screen(dataset_name: str, task: str, candidates: list[dict], force: bool = False) -> list[dict]:
    return [fit_model(dataset_name=dataset_name, task=task, force=force, **candidate) for candidate in candidates]


def results_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for run in results:
        row = {
            'dataset': run['dataset_name'],
            'task': run['task'],
            'variant': run['config'].get('variant', 'screen'),
            'hidden_dims': tuple(run['config']['hidden_dims']),
            'hidden_label': run['hidden_label'],
            'activation': run['config']['hidden_activation'],
            'optimizer': run['config']['optimizer_name'],
            'epochs': run['config']['epochs'],
            'stopped_epoch': run['stopped_epoch'],
            'best_epoch': run['best_epoch'],
            'batch_size': run['config']['batch_size'],
            'lr': run['config']['lr'],
            'weight_decay': run['config']['weight_decay'],
            'early_stopping': run['config']['early_stopping'],
            'params': run['params'],
            'seconds': run['seconds'],
        }
        if run['task'] == 'regression':
            row['val_metric'] = run['best']['val_mse_raw']
            row['test_metric'] = run['best']['test_mse_raw']
        else:
            row['val_metric'] = run['best']['val_f1']
            row['test_metric'] = run['best']['test_f1']
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if results[0]['task'] == 'regression':
        return frame.sort_values(['test_metric', 'val_metric', 'params', 'seconds']).reset_index(drop=True)
    return frame.sort_values(['test_metric', 'val_metric', 'params', 'seconds'], ascending=[False, False, True, True]).reset_index(drop=True)


def select_best_architecture(results: list[dict]) -> dict:
    if results[0]['task'] == 'regression':
        ordered = sorted(results, key=lambda run: (run['best']['val_mse_raw'], run['best']['test_mse_raw'], run['params']))
    else:
        ordered = sorted(results, key=lambda run: (-run['best']['val_f1'], -run['best']['test_f1'], run['params']))
    return ordered[0]

def plot_learning_curves(results: list[dict], title: str, task: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for run in results:
        label = run['config'].get('variant', run['hidden_label'])
        axes[0].plot(run['history']['epoch'], run['history']['train_loss'], linewidth=2, label=label)
        axes[0].plot(run['history']['epoch'], run['history']['val_loss'], linewidth=1.6, linestyle='--', alpha=0.9)
        axes[1].plot(run['history']['epoch'], run['history']['score'], linewidth=2, label=label)
    axes[0].set_title(f'{title}: train i val loss')
    axes[0].set_xlabel('epoka')
    axes[0].set_ylabel('loss')
    axes[1].set_title(f'{title}: walidacja')
    axes[1].set_xlabel('epoka')
    axes[1].set_ylabel('MSE raw' if task == 'regression' else 'macro F1')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_variant_ranking(frame: pd.DataFrame, title: str, task: str):
    ordered = frame.copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    labels = ordered['variant']
    axes[0].bar(labels, ordered['test_metric'], color='#355C7D')
    axes[0].set_title(f'{title}: wynik testowy')
    axes[0].set_ylabel('MSE' if task == 'regression' else 'macro F1')
    scatter = axes[1].scatter(ordered['seconds'], ordered['test_metric'], c=ordered['weight_decay'], s=120, cmap='viridis')
    for row in ordered.itertuples():
        axes[1].annotate(row.variant, (row.seconds, row.test_metric), fontsize=9, xytext=(5, 5), textcoords='offset points')
    axes[1].set_title(f'{title}: wynik vs czas')
    axes[1].set_xlabel('sekundy')
    axes[1].set_ylabel('MSE' if task == 'regression' else 'macro F1')
    fig.colorbar(scatter, ax=axes[1], label='weight decay')
    plt.tight_layout()
    plt.show()


def plot_regression_fit(run: dict):
    data = get_dataset(run['dataset_name'], 'regression', val_ratio=run['config']['val_ratio'])
    x_min = min(data['x_train_raw'].min(), data['x_val_raw'].min(), data['x_test_raw'].min())
    x_max = max(data['x_train_raw'].max(), data['x_val_raw'].max(), data['x_test_raw'].max())
    x_grid_raw = np.linspace(float(x_min), float(x_max), 500).reshape(-1, 1)
    x_grid = (x_grid_raw - data['x_mean']) / data['x_std']
    y_grid_raw = run['model'].forward(x_grid) * data['y_std'] + data['y_mean']
    plt.figure(figsize=(12, 5))
    plt.scatter(data['x_train_raw'], data['y_train_raw'], s=25, alpha=0.55, color='#355C7D', label='train')
    plt.scatter(data['x_val_raw'], data['y_val_raw'], s=38, alpha=0.75, color='#C06C84', label='val')
    plt.scatter(data['x_test_raw'], data['y_test_raw'], s=18, alpha=0.35, color='#6C5B7B', label='test')
    plt.plot(x_grid_raw, y_grid_raw, color='#F67280', linewidth=2.5, label='predykcja')
    plt.title(f"{run['dataset_name']}: {run['config']['variant']}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def plot_regression_residuals(run: dict):
    data = get_dataset(run['dataset_name'], 'regression', val_ratio=run['config']['val_ratio'])
    residuals = run['best']['test_pred_raw'].ravel() - data['y_test_raw'].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].scatter(data['x_test_raw'].ravel(), residuals, s=18, alpha=0.7, color='#C06C84')
    axes[0].axhline(0.0, color='black', linewidth=1)
    axes[0].set_title(f"{run['dataset_name']}: residuals")
    axes[1].hist(residuals, bins=25, color='#355C7D', alpha=0.85)
    axes[1].set_title(f"{run['dataset_name']}: rozkład błędów")
    plt.tight_layout()
    plt.show()


def plot_confusion(run: dict):
    data = get_dataset(run['dataset_name'], 'classification', val_ratio=run['config']['val_ratio'])
    matrix = run['best']['confusion_matrix']
    labels = data['class_names']
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('pred')
    ax.set_ylabel('true')
    ax.set_title(f"{run['dataset_name']}: {run['config']['variant']}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_decision_map(run: dict):
    data = get_dataset(run['dataset_name'], 'classification', val_ratio=run['config']['val_ratio'])
    x_all = np.vstack([data['x_train_raw'], data['x_val_raw'], data['x_test_raw']])
    x0_min, x1_min = x_all.min(axis=0)
    x0_max, x1_max = x_all.max(axis=0)
    pad0 = 0.05 * (x0_max - x0_min)
    pad1 = 0.05 * (x1_max - x1_min)
    xs = np.linspace(x0_min - pad0, x0_max + pad0, 240)
    ys = np.linspace(x1_min - pad1, x1_max + pad1, 240)
    xx, yy = np.meshgrid(xs, ys)
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = (grid_raw - data['x_mean']) / data['x_std']
    pred_idx = np.argmax(run['model'].forward(grid_scaled), axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, pred_idx, levels=np.arange(len(data['class_names']) + 1) - 0.5, alpha=0.28, cmap='Accent')
    for idx, class_name in enumerate(data['class_names']):
        mask = data['y_test_labels'] == class_name
        ax.scatter(data['x_test_raw'][mask, 0], data['x_test_raw'][mask, 1], s=18, alpha=0.8, color=CLASS_COLORS[idx % len(CLASS_COLORS)], label=class_name, edgecolor='black', linewidth=0.2)
    ax.set_title(f"{run['dataset_name']}: {run['config']['variant']}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.tight_layout()
    plt.show()


__all__ = [name for name in globals() if not name.startswith('_')]
