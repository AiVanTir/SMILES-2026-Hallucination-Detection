# Solution Report

## 1. Reproducibility

### Environment

The solution was developed and tested with Python 3.12 in a virtual environment.

To reproduce the result, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python solution.py
````

Running `solution.py` produces:

* `results.json`
* `predictions.csv`

No changes to the fixed infrastructure files are required.

## 2. Final Solution

The final solution modifies only the student-editable components:

* `aggregation.py`
* `probe.py`
* `splitting.py`

The fixed evaluation pipeline in `solution.py` is left unchanged.

### Hidden-state aggregation

The baseline used the last real token from the final transformer layer. I performed a layer ablation study and found that the third-to-last layer provided a stronger representation than the final layer.

The final aggregation uses a weighted combination of the third-to-last and final transformer layers:

```python
raw_feature = 0.75 * h[-3] + 0.25 * h[-1]
```

Then I concatenate this raw representation with its L2-normalized version:

```python
norm_feature = raw_feature / (||raw_feature|| + 1e-6)
feature = concat(raw_feature, norm_feature)
```

This keeps both magnitude-based and direction-based information.

### Probe classifier

The final probe keeps the lightweight MLP structure, but adds a small dropout layer:

```python
Linear(input_dim, 256)
ReLU
Dropout(0.10)
Linear(256, 1)
```

The random seed is fixed for reproducibility.

### Splitting

I kept the original stratified train/validation/test split. Alternative k-fold and reduced-holdout strategies were tested but did not improve the internal evaluation.

## 3. Experiments

### Baseline

The baseline used the last real token from the final transformer layer and a small MLP probe.

### Layer ablation

I tested several hidden layers:

| Layer               | Result            |
| ------------------- | ----------------- |
| `hidden_states[-1]` | weaker than `-3`  |
| `hidden_states[-2]` | weaker            |
| `hidden_states[-3]` | best single layer |
| `hidden_states[-4]` | weaker            |
| `hidden_states[-6]` | weaker            |
| `hidden_states[-8]` | weaker            |

The third-to-last layer was selected as the strongest single-layer representation.

### Layer fusion

I tested weighted combinations of the best layer and the final layer. The best trade-off was:

```python
0.75 * hidden_states[-3] + 0.25 * hidden_states[-1]
```

### Raw + normalized representation

Concatenating the raw weighted vector with its L2-normalized version improved the internal accuracy. This suggests that both activation magnitude and representation direction contain useful information.

### Regularization

A small dropout layer improved secondary metrics without reducing accuracy. Larger architectural changes did not help.

## 4. Failed Attempts

The following ideas were tested but discarded:

* Logistic Regression on baseline hidden-state features.
* PCA + Logistic Regression.
* RBF-SVM.
* MLP ensemble.
* Checkpoint selection by validation metric.
* Mean pooling over the last 16 tokens.
* Concatenating several full layers without feature control.
* SelectKBest feature selection after multi-layer aggregation.
* Geometric scalar features.
* Activation clipping as the final solution.
* Stratified k-fold splitting.

Most of these approaches either reduced internal accuracy or caused unstable validation/test behavior.

## 5. Final Result

The final solution achieved better internal accuracy than the original baseline while remaining simple, reproducible, and lightweight.

The main contributing changes were:

1. selecting a more informative hidden layer;
2. combining the third-to-last and final layer representations;
3. concatenating raw and L2-normalized features;
4. adding light dropout regularization.
