# PathMNIST Federated Learning Experiment Log

## Project Setup

- Dataset: MedMNIST `PathMNIST`
- Model: ResNet18, modified for 28x28 RGB images
- Clients: 10
- Partition: IID
- Framework/API: Flower `NumPyClient`
- Implemented methods:
  - Sync FedAvg
  - Naive Async
  - Staleness-aware Async
- Project folder: `pathMNIST/`
- Main entrypoint: `pathMNIST/src/fed_pathmnist/run.py`

## How To Run

From the `pathMNIST/` folder:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Then run with the CLI installed by `pyproject.toml`:

```bash
fed-pathmnist --method sync_fedavg --rounds 100 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --device cuda
```

Equivalent module form:

```bash
python -m fed_pathmnist.run --method sync_fedavg --rounds 100 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --device cuda
```

Direct file form:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --method sync_fedavg --rounds 100 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --device cuda
```

## Round vs Epoch

In federated learning, one round is not the same as one epoch.

- One federated round means:
  1. Server sends the current global model to clients.
  2. Each client trains locally.
  3. Clients return model updates.
  4. Server aggregates updates.
- `local_epochs=1` means each client trains for one local epoch on its own local partition during each federated round.
- With 10 IID clients and full PathMNIST, each client gets about one tenth of the training set.

## Data

The code uses MedMNIST's official `PathMNIST` dataset:

```python
from medmnist import INFO, PathMNIST
```

Full dataset counts observed during the run:

```text
train_examples=89996
test_examples=7180
```

Downloaded dataset location during local testing:

```text
data/medmnist/pathmnist.npz
```

The dataset file is intentionally ignored by git.

## CUDA Environment

The installed PyTorch was CUDA-enabled:

```text
torch 2.11.0+cu130
cuda_available True
device0 NVIDIA GeForce RTX 5090
```

The training commands were run with:

```bash
--device cuda
```

## Implemented Learning Rate Scheduler

Added CLI options:

```bash
--lr-scheduler none|cosine|step
--min-lr 0.0001
--step-size 30
--gamma 0.1
```

The 100-round full dataset run used cosine scheduling:

```bash
--lr 0.01 --lr-scheduler cosine --min-lr 0.0001
```

The scheduler is applied per federated round for Sync FedAvg and per async event for async methods.

## Data Augmentation

Added train-only augmentation through:

```bash
--augment
```

Applied only to the train split:

- Random horizontal flip
- Random vertical flip
- Random rotation, 15 degrees
- Random affine translation, 5%
- Mild color jitter
- Normalize with MedMNIST PathMNIST mean/std

The test split does not use random augmentation. It only uses tensor conversion and normalization.

## Sync FedAvg Results

### Smoke Test

Small PathMNIST subset:

```bash
fed-pathmnist --method sync_fedavg --rounds 1 --clients 10 --max-train-samples 200 --max-test-samples 100 --batch-size 32 --device cuda
```

Result:

```text
round=1 train_loss=2.2856 test_loss=2.1433 test_acc=0.1800
```

This was only a pipeline check, not a meaningful performance result.

### 50 Rounds, Subset, No Augmentation

Command:

```bash
fed-pathmnist --method sync_fedavg --rounds 50 --clients 10 --max-train-samples 10000 --max-test-samples 2000 --batch-size 128 --lr 0.01 --local-epochs 1 --device cuda
```

Observed result:

```text
best:  round 41, test_acc=0.8340
final: round 50, test_acc=0.8040
```

### 100 Rounds, Full Dataset, Cosine LR, No Augmentation

This run was interrupted at user request after augmentation was requested.

Command:

```bash
fed-pathmnist --method sync_fedavg --rounds 100 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --device cuda
```

Observed before interruption:

```text
train_examples=89996
test_examples=7180
best observed: round 51, test_acc=0.8628
```

The run continued briefly while being interrupted and printed through about round 81, but the best observed value remained `0.8628`.

### 100 Rounds, Full Dataset, Cosine LR, With Augmentation

Command:

```bash
fed-pathmnist --method sync_fedavg --rounds 100 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --device cuda
```

Observed result:

```text
train_examples=89996
test_examples=7180
best:  round 65, test_acc=0.9033
final: round 100, test_acc=0.8961
```

The augmentation run improved over the no-augmentation baseline and reduced overfitting. In the no-augmentation run, train loss dropped near zero while test accuracy was unstable. With augmentation, train loss remained higher and test accuracy reached about 90%.

## Async Runs

Async methods use events instead of rounds. One async event means one client update arrives and is applied to the server model. For a rough compute comparison with Sync FedAvg, 50 Sync FedAvg rounds with 10 clients correspond to about 500 async events.

Both full-dataset async methods have been run for 500 events.

Summary:

| Method | Events | Best Test Acc | Final Test Acc |
| --- | ---: | ---: | ---: |
| Naive Async | 500 | 0.8854 | 0.8777 |
| Staleness-aware Async | 500 | 0.8572 | 0.8565 |

### Naive Async Smoke Test

Command:

```bash
fed-pathmnist --method naive_async --events 3 --clients 5 --max-train-samples 100 --max-test-samples 50 --batch-size 25 --eval-every 1 --device cuda
```

Observed output:

```text
event=1 method=naive_async cid=1 staleness=0 alpha=0.5000 train_loss=2.3092 test_loss=2.1940 test_acc=0.1400
event=2 method=naive_async cid=3 staleness=1 alpha=0.5000 train_loss=2.2694 test_loss=2.2005 test_acc=0.1400
event=3 method=naive_async cid=2 staleness=2 alpha=0.5000 train_loss=2.2844 test_loss=2.2082 test_acc=0.1400
```

### Staleness-Aware Async Smoke Test

Command:

```bash
fed-pathmnist --method staleness_async --events 3 --clients 5 --max-train-samples 100 --max-test-samples 50 --batch-size 25 --eval-every 1 --device cuda
```

Observed output:

```text
event=1 method=staleness_async cid=1 staleness=0 alpha=0.5000 train_loss=2.3092 test_loss=2.1940 test_acc=0.1400
event=2 method=staleness_async cid=3 staleness=1 alpha=0.2500 train_loss=2.2694 test_loss=2.1974 test_acc=0.1400
event=3 method=staleness_async cid=2 staleness=2 alpha=0.1667 train_loss=2.2844 test_loss=2.2009 test_acc=0.1400
```

The smoke tests confirm that async event scheduling, stale update tracking, and staleness-aware alpha decay are functioning. They are not meaningful accuracy comparisons because the sample count and event count are intentionally tiny.

### Naive Async, 500 Events, Full Dataset, Cosine LR, With Augmentation

This run is approximately comparable to 50 Sync FedAvg rounds in terms of total client updates:

```text
50 sync rounds * 10 clients = 500 client updates
```

Command:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --method naive_async --events 500 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 25 --device cuda
```

Observed result:

```text
train_examples=89996
test_examples=7180
best:  event 325, test_acc=0.8854
final: event 500, test_acc=0.8777
```

Selected checkpoints:

```text
event=25   test_acc=0.6816
event=50   test_acc=0.7475
event=75   test_acc=0.8084
event=100  test_acc=0.6337
event=150  test_acc=0.8409
event=175  test_acc=0.8539
event=250  test_acc=0.8806
event=325  test_acc=0.8854
event=400  test_acc=0.8772
event=500  test_acc=0.8777
```

Observation: naive async can improve substantially, but it is less stable than Sync FedAvg. A clear example is event 100, where the test accuracy dropped to `0.6337`. This is expected because naive async applies stale updates with a fixed `alpha=0.5`, even when staleness is high.

### Staleness-Aware Async, 500 Events, Full Dataset, Cosine LR, With Augmentation

This run is also approximately comparable to 50 Sync FedAvg rounds in terms of total client updates:

```text
50 sync rounds * 10 clients = 500 client updates
```

Command:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --method staleness_async --events 500 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 25 --device cuda
```

Observed result:

```text
train_examples=89996
test_examples=7180
best:  event 400, test_acc=0.8572
final: event 500, test_acc=0.8565
```

Selected checkpoints:

```text
event=25   staleness=12 alpha=0.0385 test_acc=0.3465
event=50   staleness=9  alpha=0.0500 test_acc=0.6040
event=75   staleness=4  alpha=0.1000 test_acc=0.7568
event=100  staleness=2  alpha=0.1667 test_acc=0.7942
event=125  staleness=8  alpha=0.0556 test_acc=0.8011
event=150  staleness=15 alpha=0.0312 test_acc=0.7943
event=175  staleness=13 alpha=0.0357 test_acc=0.7948
event=200  staleness=3  alpha=0.1250 test_acc=0.8338
event=250  staleness=13 alpha=0.0357 test_acc=0.8453
event=275  staleness=11 alpha=0.0417 test_acc=0.8455
event=300  staleness=4  alpha=0.1000 test_acc=0.8499
event=350  staleness=3  alpha=0.1250 test_acc=0.8558
event=400  staleness=10 alpha=0.0455 test_acc=0.8572
event=450  staleness=5  alpha=0.0833 test_acc=0.8565
event=500  staleness=4  alpha=0.1000 test_acc=0.8565
```

Observation: staleness-aware async was more conservative because the effective alpha was reduced as staleness increased. It did not reach the same best accuracy as naive async within 500 events, but its curve was smoother and did not show a severe collapse like naive async at event 100.

## Suggested Next Experiments

Run longer full-dataset async experiments with augmentation and cosine LR:

```bash
fed-pathmnist --method naive_async --events 1000 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 20 --device cuda
```

```bash
fed-pathmnist --method staleness_async --events 1000 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 20 --device cuda
```

Async uses events instead of rounds. One event means one client update arrives and is applied to the server model.

For fair comparison, choose an async event count that roughly matches the amount of client training in Sync FedAvg. Since one Sync FedAvg round with 10 clients uses 10 client updates, 100 Sync FedAvg rounds correspond roughly to 1000 async events.
