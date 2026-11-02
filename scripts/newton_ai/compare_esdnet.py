"""Diagnostic screen-demoireing baseline on rendered RGB; NOT a RAW correction model.
Use the inspected official architecture at the revision documented in docs/newton-ai.md.
"""

import argparse, hashlib, importlib.util, json, pathlib, time
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--reference", type=pathlib.Path, required=True)
p.add_argument("--weights", type=pathlib.Path, required=True)
p.add_argument("--images", type=pathlib.Path, required=True)
p.add_argument("--output", type=pathlib.Path, required=True)
p.add_argument(
    "--comparison-triptychs",
    action="store_true",
    help="Use first third of OpenRevelare verification JPEGs",
)
a = p.parse_args()
sha = hashlib.sha256(a.weights.read_bytes()).hexdigest()
if sha != "bdcdd33f11e1d5eb836671f15991ecb42134bd5ba98c1e4de3b8e2f4138fdb2b":
    raise ValueError("Expected the documented official mixed-dataset checkpoint")
spec = importlib.util.spec_from_file_location("esdnet", a.reference / "model/nets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = module.my_model(48, 32, 64, 32, 1).eval().to(device)
model.load_state_dict(
    torch.load(a.weights, map_location="cpu", weights_only=True), strict=True
)
a.output.mkdir(parents=True, exist_ok=True)
report = []
for path in sorted(a.images.glob("*.jpg")):
    with Image.open(path) as source:
        im = source.convert("RGB")
    if a.comparison_triptychs:
        im = im.crop((0, 0, im.width // 3, im.height))
    im.thumbnail((1024, 1024))
    x = (
        torch.from_numpy(np.array(im).copy())
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
        .to(device)
    )
    h, w = x.shape[2:]
    pw = (-w) % 32
    ph = (-h) % 32
    padded = torch.cat(
        [
            F.pad(
                x[:, c : c + 1], (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=v
            )
            for c, v in enumerate([0.3827, 0.4141, 0.3912])
        ],
        1,
    )
    start = time.monotonic()
    with torch.inference_mode():
        y = (
            model(padded)[0][:, :, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
            .clamp(0, 1)
            .cpu()
        )
    before = np.array(im)
    after = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    result = Image.new("RGB", (w * 2, h))
    result.paste(im, (0, 0))
    result.paste(Image.fromarray(after), (w, 0))
    result.save(a.output / path.name)
    report.append(
        dict(
            file=path.name,
            seconds=time.monotonic() - start,
            device=device,
            mean_abs_change=float(np.abs(before.astype(float) - after).mean() / 255),
        )
    )
(a.output / "report.json").write_text(
    json.dumps(
        dict(weights_sha256=sha, diagnostic_rendered_rgb=True, results=report), indent=2
    )
)
