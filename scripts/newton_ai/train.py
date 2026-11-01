"""Local synthetic-degradation training. Target real frames 1934..1937 are never training data."""

import argparse, hashlib, json, math, pathlib, struct, time
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.nn import functional as F
from model import FringeNet

TARGETS = {f"DSC019{i}.rgbf" for i in range(34, 38)}
VALIDATION = {f"DSC019{i}.rgbf" for i in range(30, 34)}


def read_frame(path):
    with path.open("rb") as stream:
        w, h = struct.unpack("<ii", stream.read(8))
        data = np.frombuffer(stream.read(), dtype="<f4").reshape(h, w, 3)
    if w < 192 or h < 192:
        raise ValueError(f"Image is too small: {path}")
    return data


def batch(frames, rng, count=6, size=192):
    xx, yy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    features = []
    targets = []
    masks = []
    for _ in range(count):
        frame = frames[int(rng.integers(len(frames)))]
        h, w = frame.shape[:2]
        x = int(rng.integers(max(1, w - size)))
        y = int(rng.integers(max(1, h - size)))
        clean = np.log(
            np.maximum(frame[y : y + size, x : x + size].astype(np.float32), 1e-5)
        )
        clean = np.rot90(clean, int(rng.integers(4))).copy()
        if rng.random() < 0.5:
            clean = clean[:, ::-1].copy()
        clean = (clean - clean.mean((0, 1), keepdims=True)) * rng.uniform(0.6, 1.4)
        if rng.random() < 0.35:
            # Include neutral/coloured smooth scenes and edges: the network must learn
            # monochrome interference without relying on opponent colour evidence.
            clean = np.empty((size, size, 3), np.float32)
            shared = (
                rng.uniform(-0.6, 0.6) * xx
                + rng.uniform(-0.6, 0.6) * yy
                + 0.06 * np.sin(3 * xx + 4 * yy)
            )
            if rng.random() < 0.5:
                shared += (xx > rng.uniform(-0.7, 0.7)) * rng.uniform(-0.35, 0.35)
            for c in range(3):
                clean[:, :, c] = (
                    shared
                    + rng.uniform(-0.12, 0.12) * xx
                    + rng.uniform(-0.12, 0.12) * yy
                )
            clean += rng.normal(0, 0.006, clean.shape)
        residual = np.zeros_like(clean)
        mask = np.zeros((size, size), np.float32)
        if rng.random() > 0.4:
            theta = rng.uniform(-math.pi, math.pi)
            u = xx * math.cos(theta) + yy * math.sin(theta)
            v = -xx * math.sin(theta) + yy * math.cos(theta)
            kind = int(rng.integers(5))
            freq = rng.uniform(4, 80)
            if kind == 0:
                phase = (
                    freq * u
                    + rng.uniform(-16, 16) * v * v
                    + rng.uniform(-12, 12) * u * v
                )
            elif kind == 1:
                phase = freq * np.sqrt(
                    (u - rng.uniform(-1, 1)) ** 2 * rng.uniform(0.3, 1.8)
                    + (v - rng.uniform(-1, 1)) ** 2
                    + rng.uniform(0.01, 0.1)
                )
            elif kind == 2:
                phase = (
                    freq * u
                    + rng.uniform(3, 14) * np.sin(v * rng.uniform(1, 5))
                    + rng.uniform(-10, 10) * u * u
                )
            elif kind == 3:
                phase = freq * (u * u - v * v) * 0.6 + rng.uniform(3, 15) * u
            else:
                phase = (
                    freq * u
                    + rng.uniform(-12, 12) * u * u
                    + rng.uniform(-12, 12) * v**3
                )
            phase += rng.uniform(-math.pi, math.pi)
            if rng.random() < 0.55:
                knots = rng.normal(size=(5, 5))
                field = zoom(knots, size / 5, order=3)[:size, :size]
                mask = (1 / (1 + np.exp(-field * 4))).astype(np.float32)
            else:
                mask.fill(1)
            alpha = rng.uniform(0.025, 0.19)
            monochrome = rng.random() < 0.45
            factor = (
                np.ones(3)
                if monochrome
                else np.array([550 / 650, 1, 550 / 450]) * rng.uniform(0.9, 1.1, 3)
            )
            offsets = np.zeros(3) if monochrome else rng.uniform(-math.pi, math.pi, 3)
            for c in range(3):
                residual[:, :, c] = (
                    mask * alpha * np.cos(phase * factor[c] + offsets[c])
                )
            if rng.random() < 0.25:
                for c in range(3):
                    residual[:, :, c] += (
                        0.035 * mask * np.sin(u * freq * 0.7 + v * freq * 0.8 + c * 1.2)
                    )
            residual -= residual.mean((0, 1), keepdims=True)
        observed = clean + residual
        observed = (observed - observed.mean((0, 1), keepdims=True)) / 4
        features.append(observed.transpose(2, 0, 1))
        targets.append(residual.transpose(2, 0, 1))
        masks.append(mask[None])
    return tuple(
        torch.from_numpy(np.stack(a).astype(np.float32))
        for a in (features, targets, masks)
    )


def evaluate(model, frames, device):
    rng = np.random.default_rng(498217)
    records = []
    model.eval()
    with torch.inference_mode():
        for _ in range(4):
            x, target, mask = batch(frames, rng, 8)
            prediction = model(x.to(device)).cpu()
            residual = prediction[:, :3]
            confidence = prediction[:, 3:4]
            gated = residual * ((confidence - 0.15) / 0.35).clamp(0, 1)
            active = target.abs().mean((1, 2, 3)) > 0.001
            before = (target[active] ** 2).mean().item()
            after = ((target - gated)[active] ** 2).mean().item()
            clean = ~active
            false = gated[clean].abs().mean().item() if clean.any() else 0
            clean_mse = (gated[clean] ** 2).mean().item() if clean.any() else 0
            records.append(
                dict(
                    before_mse=before,
                    after_mse=after,
                    clean_mean_abs_log_change=false,
                    clean_mse=clean_mse,
                )
            )
    model.train()
    return {key: float(np.mean([r[key] for r in records])) for key in records[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=6400)
    parser.add_argument("--seed", type=int, default=64821)
    parser.add_argument("--validation", nargs="+", default=sorted(VALIDATION))
    parser.add_argument("--exclude", nargs="*", default=sorted(TARGETS))
    args = parser.parse_args()
    validation = set(args.validation)
    exclude = set(args.exclude)
    if validation & exclude or args.steps < 1:
        raise ValueError(
            "Validation and exclusions must be disjoint; steps must be positive."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.data.glob("*.rgbf"))
    train_paths = [p for p in paths if p.name not in exclude | validation]
    valid_paths = [p for p in paths if p.name in validation]
    if not train_paths or not valid_paths:
        raise ValueError("Training and image-disjoint validation frames are required.")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = FringeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.steps, eta_min=5e-5
    )
    train = [read_frame(p) for p in train_paths]
    valid = [read_frame(p) for p in valid_paths]
    manifest = dict(
        seed=args.seed,
        steps=args.steps,
        device=device,
        torch=torch.__version__,
        training=[
            dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in train_paths
        ],
        validation=[p.name for p in valid_paths],
        excluded_targets=sorted(exclude),
    )
    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2))
    best = float("inf")
    start = time.monotonic()
    history = []
    for step in range(1, args.steps + 1):
        x, target, mask = batch(train, rng)
        x = x.to(device).contiguous()
        target = target.to(device).contiguous()
        mask = mask.to(device).contiguous()
        pred = model(x)
        delta = pred[:, :3].contiguous()
        loss = 4 * F.smooth_l1_loss(
            delta, target, beta=0.025
        ) + 0.025 * F.binary_cross_entropy(pred[:, 3:4].contiguous(), mask.contiguous())
        loss += 0.3 * (
            F.l1_loss(
                delta[:, :, 1:] - delta[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]
            )
            + F.l1_loss(
                delta[:, :, :, 1:] - delta[:, :, :, :-1],
                target[:, :, :, 1:] - target[:, :, :, :-1],
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            print(
                json.dumps(
                    dict(
                        step=step,
                        loss=float(loss.detach().cpu()),
                        seconds=round(time.monotonic() - start, 1),
                    )
                ),
                flush=True,
            )
        if step % 200 == 0 or step == args.steps:
            score = evaluate(model, valid, device)
            score.update(step=step)
            history.append(score)
            print(json.dumps(dict(validation=score)), flush=True)
            if score["after_mse"] + 2 * score["clean_mse"] < best:
                best = score["after_mse"] + 2 * score["clean_mse"]
                torch.save(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    args.output / "fringenet.pth",
                )
            (args.output / "validation.json").write_text(json.dumps(history, indent=2))
    print("Finished", device, flush=True)


if __name__ == "__main__":
    main()
