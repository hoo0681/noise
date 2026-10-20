"""CutBI against CutBIR (its measurements reordered) on one sequence, one GPU: agreement with each other and with the
gradient (autograd, for this check only), and the time of one estimate. Not part of the submission.

  CUDA_VISIBLE_DEVICES=0 python check_reorder.py WEIGHTS.pt [--route-k K]
"""
import argparse
import importlib.util
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('track1', ROOT / 'train/val/run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

BLOCK = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')


def group(name):
    if name.startswith('transformer.h.'):
        return name.split('.', 3)[3].removesuffix('.weight')
    return name.split('.')[0] if name.startswith('value_embeds') else name.removesuffix('.weight')


def cos(a, b):
    return ((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def table(estimates, reference):
    groups = {}
    for name, ref in reference.items():
        groups.setdefault(group(name), []).append(name)
    rows = {}
    for g, names in groups.items():
        cat = lambda grads: torch.cat([grads[n].float().flatten() for n in names])
        rows[g] = [cos(cat(e), cat(reference)) for e in estimates]
    block = [n for n in reference if n.startswith('transformer.h.') and group(n) in BLOCK]
    cat = lambda grads: torch.cat([grads[n].float().flatten() for n in block])
    rows['ALL BLOCK MATRICES'] = [cos(cat(e), cat(reference)) for e in estimates]
    return rows


def timed(estimator, x, y, repeats=3):
    estimator.estimate(x, y)  # compile
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        out = estimator.estimate(x, y)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=Path)
    parser.add_argument('--route-k', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    config = run.GPTConfig(n_layer=8, n_head=8, n_kv_head=8, n_embd=512)
    settings = run.Settings(fc_k=run.HIDDEN // 8, route_k=args.route_k or run.HIDDEN // 16)
    model = run.make_model(device, 0, config)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in state.items()})
    row = run.load_split(ROOT / 'data/val.pt', run.VAL_ROWS)[:1].to(device)
    x, y = row[:, :-1].contiguous(), row[:, 1:].contiguous()

    for p in model.parameters():
        p.requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x, y, loss_reduction='sum')
    true = dict(zip([n for n, _ in model.named_parameters()],
                    [g.float() for g in torch.autograd.grad(loss, list(model.parameters()))]))
    model.requires_grad_(False)

    results, times = [], []
    for cls in (run.CutBI, run.CutBIR):
        estimator = cls(model, settings, 0, 0, 1)
        estimate, seconds = timed(estimator, x, y)
        estimator.close()
        results.append({n: g.float() for n, g in estimate.items()})
        times.append(seconds)
    rows = table(results, true)
    between = table([results[1]], results[0])
    print(f'weights {args.weights}, route_k {settings.route_k}, 1 sequence of {x.shape[1]} tokens')
    print(f"{'group':<22}{'CutBI~grad':>12}{'CutBIR~grad':>13}{'CutBIR~CutBI':>14}")
    for g, (a, b) in rows.items():
        print(f'{g:<22}{a:>12.4f}{b:>13.4f}{between[g][0]:>14.4f}')
    print(f'estimate time: CutBI {times[0]:.3f} s, CutBIR {times[1]:.3f} s ({times[0] / times[1]:.2f}x)')


if __name__ == '__main__':
    main()
