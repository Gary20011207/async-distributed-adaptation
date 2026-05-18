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

Async methods use events instead of rounds. One async event means one client update arrives and is applied to the server model. For rough compute comparison:

```text
50 Sync FedAvg rounds * 10 clients = 500 async events
100 Sync FedAvg rounds * 10 clients = 1000 async events
```

Both full-dataset async methods have been run for 500 and 1000 events.

Summary:

| Method | Events | Best Test Acc | Final Test Acc |
| --- | ---: | ---: | ---: |
| Naive Async | 500 | 0.8854 | 0.8777 |
| Staleness-aware Async | 500 | 0.8572 | 0.8565 |
| Naive Async | 1000 | 0.9003 | 0.8903 |
| Staleness-aware Async | 1000 | 0.8830 | 0.8791 |

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

### Naive Async, 1000 Events, Full Dataset, Cosine LR, With Augmentation

This run is approximately comparable to 100 Sync FedAvg rounds in terms of total client updates:

```text
100 sync rounds * 10 clients = 1000 client updates
```

Command:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --method naive_async --events 1000 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 50 --device cuda
```

Observed result:

```text
train_examples=89996
test_examples=7180
best:  event 700, test_acc=0.9003
final: event 1000, test_acc=0.8903
```

Selected checkpoints:

```text
event=50    staleness=9  alpha=0.5000 test_acc=0.8387
event=100   staleness=2  alpha=0.5000 test_acc=0.7609
event=150   staleness=15 alpha=0.5000 test_acc=0.8014
event=200   staleness=3  alpha=0.5000 test_acc=0.8393
event=250   staleness=13 alpha=0.5000 test_acc=0.8602
event=300   staleness=4  alpha=0.5000 test_acc=0.8593
event=350   staleness=3  alpha=0.5000 test_acc=0.8730
event=400   staleness=10 alpha=0.5000 test_acc=0.8869
event=450   staleness=5  alpha=0.5000 test_acc=0.8864
event=500   staleness=4  alpha=0.5000 test_acc=0.8766
event=550   staleness=10 alpha=0.5000 test_acc=0.8792
event=600   staleness=6  alpha=0.5000 test_acc=0.8975
event=650   staleness=14 alpha=0.5000 test_acc=0.8866
event=700   staleness=5  alpha=0.5000 test_acc=0.9003
event=750   staleness=12 alpha=0.5000 test_acc=0.8822
event=800   staleness=10 alpha=0.5000 test_acc=0.8915
event=850   staleness=6  alpha=0.5000 test_acc=0.8921
event=900   staleness=10 alpha=0.5000 test_acc=0.8848
event=950   staleness=12 alpha=0.5000 test_acc=0.8884
event=1000  staleness=4  alpha=0.5000 test_acc=0.8903
```

Observation: with a fairer 1000-event budget, naive async nearly matched Sync FedAvg. Its best accuracy `0.9003` is close to Sync FedAvg's best `0.9033`, but the final accuracy `0.8903` remained lower and the curve was still unstable.

### Staleness-Aware Async, 1000 Events, Full Dataset, Cosine LR, With Augmentation

This run is approximately comparable to 100 Sync FedAvg rounds in terms of total client updates:

```text
100 sync rounds * 10 clients = 1000 client updates
```

Command:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --method staleness_async --events 1000 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 50 --device cuda
```

Observed result:

```text
train_examples=89996
test_examples=7180
best:  event 550, test_acc=0.8830
final: event 1000, test_acc=0.8791
```

Selected checkpoints:

```text
event=50    staleness=9  alpha=0.0500 test_acc=0.5904
event=100   staleness=2  alpha=0.1667 test_acc=0.7588
event=150   staleness=15 alpha=0.0312 test_acc=0.7864
event=200   staleness=3  alpha=0.1250 test_acc=0.8272
event=250   staleness=13 alpha=0.0357 test_acc=0.8368
event=300   staleness=4  alpha=0.1000 test_acc=0.8482
event=350   staleness=3  alpha=0.1250 test_acc=0.8682
event=400   staleness=10 alpha=0.0455 test_acc=0.8572
event=450   staleness=5  alpha=0.0833 test_acc=0.8643
event=500   staleness=4  alpha=0.1000 test_acc=0.8727
event=550   staleness=10 alpha=0.0455 test_acc=0.8830
event=600   staleness=6  alpha=0.0714 test_acc=0.8797
event=650   staleness=14 alpha=0.0333 test_acc=0.8706
event=700   staleness=5  alpha=0.0833 test_acc=0.8671
event=750   staleness=12 alpha=0.0385 test_acc=0.8753
event=800   staleness=10 alpha=0.0455 test_acc=0.8784
event=850   staleness=6  alpha=0.0714 test_acc=0.8799
event=900   staleness=10 alpha=0.0455 test_acc=0.8780
event=950   staleness=12 alpha=0.0385 test_acc=0.8788
event=1000  staleness=4  alpha=0.1000 test_acc=0.8791
```

Observation: 1000 events improved staleness-aware async substantially over its 500-event run. However, the current inverse staleness decay is likely too conservative; it improved stability but limited peak accuracy compared with naive async. The next step should tune the staleness decay strength, for example `alpha / (1 + staleness / tau)`.

### Staleness Decay Tuning

The code now supports multiple staleness decay rules:

```bash
--staleness-decay inverse|tau_inverse|floor_tau_inverse|exp|hinge
--staleness-tau 5
--min-alpha 0.05
```

Implemented alpha formulas:

```text
inverse:           alpha / (1 + staleness)
tau_inverse:       alpha / (1 + staleness / tau)
floor_tau_inverse: max(min_alpha, alpha / (1 + staleness / tau))
exp:               alpha * exp(-staleness / tau)
hinge:             alpha if staleness <= tau else alpha / (1 + staleness - tau)
```

The first tuning pass tested `tau_inverse` with `alpha=0.5`.

#### Tau Inverse, Tau = 3

Command:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --method staleness_async --events 1000 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 100 --alpha 0.5 --staleness-decay tau_inverse --staleness-tau 3 --device cuda
```

Observed result:

```text
best:  event 800, test_acc=0.8873
final: event 1000, test_acc=0.8872
```

Selected checkpoints:

```text
event=100   staleness=2  alpha=0.3000 test_acc=0.8038
event=200   staleness=3  alpha=0.2500 test_acc=0.8426
event=300   staleness=4  alpha=0.2143 test_acc=0.8696
event=400   staleness=10 alpha=0.1154 test_acc=0.8809
event=500   staleness=4  alpha=0.2143 test_acc=0.8848
event=600   staleness=6  alpha=0.1667 test_acc=0.8859
event=700   staleness=5  alpha=0.1875 test_acc=0.8851
event=800   staleness=10 alpha=0.1154 test_acc=0.8873
event=900   staleness=10 alpha=0.1154 test_acc=0.8864
event=1000  staleness=4  alpha=0.2143 test_acc=0.8872
```

#### Tau Inverse, Tau = 5

Command:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --method staleness_async --events 1000 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 100 --alpha 0.5 --staleness-decay tau_inverse --staleness-tau 5 --device cuda
```

Observed result:

```text
best:  event 1000, test_acc=0.8914
final: event 1000, test_acc=0.8914
```

Selected checkpoints:

```text
event=100   staleness=2  alpha=0.3571 test_acc=0.8146
event=200   staleness=3  alpha=0.3125 test_acc=0.8532
event=300   staleness=4  alpha=0.2778 test_acc=0.8741
event=400   staleness=10 alpha=0.1667 test_acc=0.8864
event=500   staleness=4  alpha=0.2778 test_acc=0.8865
event=600   staleness=6  alpha=0.2273 test_acc=0.8903
event=700   staleness=5  alpha=0.2500 test_acc=0.8872
event=800   staleness=10 alpha=0.1667 test_acc=0.8909
event=900   staleness=10 alpha=0.1667 test_acc=0.8905
event=1000  staleness=4  alpha=0.2778 test_acc=0.8914
```

Current tuning summary:

| Staleness Decay | Tau | Best Test Acc | Final Test Acc |
| --- | ---: | ---: | ---: |
| inverse | n/a | 0.8830 | 0.8791 |
| tau_inverse | 3 | 0.8873 | 0.8872 |
| tau_inverse | 5 | 0.8914 | 0.8914 |

Observation: weakening the staleness penalty improved staleness-aware async. `tau=5` is the best tested setting so far. It improved final accuracy from `0.8791` to `0.8914`, which is close to naive async final `0.8903`, but still below naive async best `0.9003` and Sync FedAvg best `0.9033`.

Planned but not run yet:

```text
tau_inverse tau=10
floor_tau_inverse tau=5 min_alpha=0.05
floor_tau_inverse tau=10 min_alpha=0.05
```

## Suggested Next Experiments

Run decay and alpha tuning experiments:

```bash
tau_inverse tau=10
floor_tau_inverse tau=5 min_alpha=0.05
floor_tau_inverse tau=10 min_alpha=0.05
alpha sweep: 0.1, 0.2, 0.3, 0.5
buffered async aggregation
```
