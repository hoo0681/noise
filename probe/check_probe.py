"""The probe estimator against autograd of the summed loss (float32, no autocast), one sequence. Not part of the run.

  CUDA_VISIBLE_DEVICES=0 python probe/check_probe.py WEIGHTS.pt
"""
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('track1', ROOT / 'train/val/run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def main():
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = run.GPTConfig(n_layer=8, n_head=8, n_kv_head=8, n_embd=512)
    settings = run.Settings(fc_k=run.HIDDEN // 8, route_k=run.HIDDEN // 4)
    model = run.make_model(device, 0, config)
    if len(sys.argv) > 1:
        state = torch.load(sys.argv[1], map_location=device)
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in state.items()})
    row = run.load_split(ROOT / 'data/train.pt', 1).to(device)
    x, y = row[:, :-1].contiguous(), row[:, 1:].contiguous()

    reference = run.make_model(device, 0, config).float()
    reference.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    reference.requires_grad_(True)
    params = dict(reference.named_parameters())
    loss = reference(x, y, loss_reduction='sum')
    true = dict(zip(params, torch.autograd.grad(loss, list(params.values()))))

    estimator = run.Probe(model, settings, 0, 0, 1)
    estimate = estimator.estimate(x, y)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        estimate = estimator.estimate(x, y)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / 3
    print(f'loss: model {loss.item():.4f}, probe forward {estimator.capture.clean_loss.sum().item():.4f}')
    worst = []
    for name, ref in true.items():
        a, b = estimate[name].float().flatten(), ref.float().flatten()
        rel = ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
        worst.append((rel, name))
    block = [n for n in true if n.startswith('transformer.h.')]
    cat = lambda g: torch.cat([g[n].float().flatten() for n in block])
    a, b = cat(estimate), cat(true)
    print(f'block matrices: cos {(a @ b / (a.norm() * b.norm())).item():.8f}, rel {((a - b).norm() / b.norm()).item():.2e}')
    for rel, name in sorted(worst, reverse=True)[:5]:
        print(f'  worst rel {rel:.2e}  {name}')
    print(f'one estimate (1 sequence): {seconds:.3f} s')


if __name__ == '__main__':
    main()
