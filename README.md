# TECOS: Temporally Contextualized Supervision for Dynamic Graph Anomaly Detection

TECOS (Temporally Contextualized Supervision) is a dynamic graph anomaly detection method built upon a SLADE/TGN-style temporal graph encoder. It enhances anomaly detection by incorporating temporal context into supervision, using recovery and drift signals to guide end-to-end learning over dynamic graph interactions.

## Repository Structure

```text
.
|-- main.py                    # Main entry point for TECOS experiments
|-- requirements.txt           # Python dependencies
|-- evaluation/
|   `-- evaluation.py          # Evaluation utilities
|-- model/
|   |-- SLADE_TGN.py           # SLADE/TGN-style temporal graph encoder
|   |-- contextual_supervision.py # Temporally contextualized supervision module
|   |-- temporal_attention_SLADE.py # Temporal attention layer used by the encoder
|   `-- time_encoding.py
|-- modules/
|   |-- embedding_module.py
|   |-- memory.py
|   |-- memory_updater.py
|   |-- message_function.py
|   `-- ssm.py
`-- utils/
    |-- data_processing.py     # Dataset loading and temporal split
    |-- e2e_training.py        # Training/evaluation helper functions
    |-- preprocess_data.py     # Raw data preprocessing
    |-- MI.py                  # Mutual information losses
    |-- cosine.py
    |-- dot.py
    |-- euclidean.py
    `-- utils.py               # Neighbor finder and shared utilities
```

`main.py` keeps the high-level TECOS experiment flow: argument parsing, data loading, model construction, training, evaluation, and logging. Reusable helper functions are separated into `utils/e2e_training.py`.

## Environment

Python 3.9 is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses PyTorch, PyTorch Geometric, and torch-scatter. If `torch-scatter` or `torch-geometric` fails to install directly from `requirements.txt`, install versions that match your local PyTorch and CUDA environment.

## Data

The main script expects preprocessed CSV files in the `data/` directory:

```text
data/ml_<dataset_name>.csv
```

Supported dataset names include:

- `wikipedia`
- `reddit`
- `bitcoinalpha`
- `bitcoinotc`

Raw CSV files should follow this format:

```text
source_id,destination_id,timestamp,label,features(optional)
```

After preprocessing, the model reads files with these columns:

```text
u,i,ts,label,idx
```

## Preprocessing

Place the raw dataset file under `data/`, for example:

```text
data/wikipedia.csv
```

Then run:

```bash
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
python utils/preprocess_data.py --data bitcoinalpha
python utils/preprocess_data.py --data bitcoinotc
```

The preprocessing script generates:

```text
data/ml_<dataset_name>.csv
data/ml_<dataset_name>.npy
data/ml_<dataset_name>_node.npy
```

The current TECOS pipeline mainly uses `data/ml_<dataset_name>.csv`.

## Running Experiments

Run the default experiment:

```bash
python main.py -d wikipedia
```

Run other datasets:

```bash
python main.py -d reddit
python main.py -d bitcoinalpha
python main.py -d bitcoinotc
```

Example with common options:

```bash
python main.py -d wikipedia --n_epoch 10 --n_runs 3 --bs 100
```

View all command-line options:

```bash
python main.py --help
```

## Important Arguments

| Argument                     | Default         | Description                                       |
| ---------------------------- | --------------- | ------------------------------------------------- |
| `-d`, `--data`               | `wikipedia`     | Dataset name                                      |
| `--bs`                       | `100`           | Batch size                                        |
| `--n_degree`                 | `20`            | Number of temporal neighbors                      |
| `--n_head`                   | `2`             | Number of attention heads                         |
| `--n_epoch`                  | `10`            | Number of training epochs                         |
| `--n_runs`                   | `3`             | Number of repeated runs                           |
| `--seed`                     | `0`             | Base random seed                                  |
| `--lr`                       | `3e-6`          | Main model learning rate                          |
| `--stage2_lr`                | `5e-4`          | Contextual supervision module learning rate       |
| `--training_ratio`           | `0.85`          | Temporal split ratio for training data            |
| `--mi_method`                | `mine`          | Mutual information estimator: `mine` or `infonce` |
| `--distance_metric`          | `cosine`        | Distance metric: `cosine` or `euclidean`          |
| `--label_delay`              | `0`             | Label delay applied by source node                |
| `--delay_apply_to`           | `both`          | Apply label delay to `train`, `test`, or `both`   |
| `--stage2_window_size`       | `1`             | Base temporal context size for supervision        |
| `--stage2_batch_size`        | `128`           | Batch size for supervision chunks                 |
| `--stage2_pooling_type`      | `softmax`       | Pooling strategy in the supervision module        |
| `--stage2_context_direction` | `bidirectional` | Temporal context direction for supervision        |
| `--e2e_chunk_batches`        | `8`             | Number of mini-batches per optimization chunk     |
| `--e2e_alpha`                | `1.0`           | Weight for the temporal contrastive encoder loss  |
| `--e2e_beta`                 | `1.0`           | Weight for the contextual supervision loss        |

## Outputs

Experiment logs are saved under:

```text
log/
```

The logs include:

- training loss
- anomaly detection AUC/AP
- training time
- inference time
- mean and standard deviation across repeated runs

## Submission Archive

For a code-only archive, keep:

```text
README.md
requirements.txt
main.py
evaluation/
model/
modules/
utils/
.gitignore
```

Exclude generated, local, or heavy files:

```text
.git/
.vscode/
data/
log/
__pycache__/
*.pyc
*.log
*.pt
*.pth
*.ckpt
.env
.venv/
venv/
```

If the submission requires full reproducibility, include the required `data/ml_<dataset_name>.csv` files or describe the dataset source and preprocessing steps in the accompanying report.

## Dataset Source

Wikipedia and Reddit datasets can be downloaded from the JODIE/SNAP page:

```text
http://snap.stanford.edu/jodie/
```
