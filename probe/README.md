# Probe: a reverse sweep with every nonlinear derivative measured by forward evaluations

A test case for the rule discussed in #17 and #20, not a record claim. `--estimator probe` in
[`train/val/run.py`](../train/val/run.py) (class `Probe`, added to #20's script) is backpropagation in which:

- every linear map (weight matrices, the attention probabilities' transpose `A^T`, rotary, reshapes, the residual and
  x0 mixing, the embedding gather) is transposed in closed form, as Relay takes its linear steps;
- every elementwise nonlinearity (the norm's `rsqrt` and square, softmax's `exp` and `1/S`, the head's soft cap, `log`
  and `exp`, `relu^2`, the gate's `2 sigmoid`) has its derivative measured by a central difference
  `(f(x + h) - f(x - h)) / 2h`, two forward evaluations of that scalar function, the probe Relay's `hidden_all` uses
  for `relu^2`;
- the partials of the products (`x * s`, `e * (1/S)`) are measured the same way.

No autograd and no analytic derivative of any nonlinear operation, the criterion proposed in #20. It is also, plainly,
a hand-written backward pass with finite-difference scalar derivatives, which is the point of the test.

## Results (8x H200, seed 0, torch 2.10.0+cu128)

**Gradient** ([`check_probe.py`](check_probe.py): autograd of the summed loss in float32, no autocast, `data/train.pt`
row 0, h = 1e-3, probes evaluated in float64):

| weights | cosine, all block matrices | relative error, block matrices | worst parameter, relative error |
| - | - | - | - |
| seed-0 init | 1.00000000 | 2.7e-6 | 4.8e-6 |
| Dust track 2 final weights | 1.00000000 | 1.9e-6 | 2.2e-5 (`x0_lambdas`) |
| this run's final weights | 0.99999994 | 2.5e-6 | 8.8e-6 (`x0_lambdas`) |

**Track 1** (full epoch, 610 updates, #20's optimizer, rates and schedule, no Boost phase):

| | val loss | train time |
| - | - | - |
| Relay (#20), rerun on the same node | 3.7044 | 141.4 s |
| Probe | 3.7110 | 59.3 s |
| backprop, same optimizer (#20's table, 8x H100, 3 seeds) | 3.7122 +- 0.0089 | 38 s |

Relay's rerun reproduces its posted seed-0 value (3.7044). Probe's loss is backprop's within seed noise, as it should
be. Logs: [`logs/track1_full_epoch`](logs/track1_full_epoch), [`logs/relay_same_node`](logs/relay_same_node).

**Shorter epochs** (the first N training rows, so the schedule spans N / 8 updates; `--lr-scale` multiplies both rates):

| updates | rate scale | val loss | train time |
| - | - | - | - |
| 150 | 1 | 4.863 | 14.7 s |
| 200 | 1 | 4.656 | 19.5 s |
| 250 | 1 | 4.452 | 24.4 s |
| 300 | 1 / 1.5 / 2 | 4.278 / 4.263 / 4.288 | 29.3 s |
| **320** | **1.5** | **4.175** | **31.3 s** |
| 330 | 1.5 | 4.120 | 32.3 s |
| 340 | 1.5 | 4.140 | 33.0 s |
| 350 | 1 / 1.5 | 4.078 / 4.063 | 34.1 s |

The 320-update run's log: [`logs/short_320_updates`](logs/short_320_updates). Times are on H200, not the H100 node the
README specifies.

```bash
torchrun --standalone --nproc_per_node=8 train/val/run.py --estimator probe                                   # track 1
torchrun --standalone --nproc_per_node=8 train/val/run.py --estimator probe --train-rows 2560 --lr-scale 1.5  # 4.2 in 320 updates
CUDA_VISIBLE_DEVICES=0 python probe/check_probe.py [weights.pt]                                               # the gradient check
```
