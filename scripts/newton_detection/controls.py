"""Report simple controls, including a deliberately indistinguishable printed-pattern mimic."""

import argparse, json, pathlib
import numpy as np
from detect import evidence

p = argparse.ArgumentParser()
p.add_argument("--output", type=pathlib.Path, required=True)
a = p.parse_args()
y, x = np.mgrid[:128, :160]
flat = np.full((128, 160, 3), 0.3, np.float32)
step = flat.copy()
step[:64] = [0.25, 0.4, 0.32]
step[64:] = [0.5, 0.28, 0.2]
checker = np.where(
    ((x // 8 + y // 8) % 2)[:, :, None], [0.25, 0.4, 0.32], [0.5, 0.28, 0.2]
).astype(np.float32)
periodic = np.stack(
    [0.3 * np.exp(0.07 * np.sin(y * 2 * np.pi / 8 + c * 1.5)) for c in range(3)], axis=2
).astype(np.float32)
noise = (
    flat * np.exp(np.random.default_rng(1417).normal(0, 0.015, flat.shape))
).astype(np.float32)
controls = {
    "flat": flat,
    "single_colour_edge": step,
    "two_colour_checkerboard": checker,
    "independent_grain": noise,
    "synthetic_interference": periodic,
    "identical_printed_pattern_mimic": periodic.copy(),
}
report = {}
for name, image in controls.items():
    maps = evidence(image)
    report[name] = {
        key: dict(
            mean=float(maps[key].mean()),
            p95=float(np.quantile(maps[key], 0.95)),
            fraction_above_015=float(np.mean(maps[key] > 0.15)),
        )
        for key in ("periodicity", "phase_chromatic")
    }
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(
    json.dumps(
        {
            "warning": "Synthetic controls only, not estimates of real-world precision. The printed mimic is intentionally pixel-identical to the interference example.",
            "controls": report,
        },
        indent=2,
    )
)
print(json.dumps(report, indent=2))
