"""Export a local residual checkpoint and verify ONNX against PyTorch."""

import argparse, hashlib, json, pathlib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from model import FringeNet

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=pathlib.Path, required=True)
p.add_argument("--output", type=pathlib.Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
model = FringeNet().eval()
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True))
x = torch.from_numpy(
    np.random.default_rng(7415).normal(0, 0.3, (1, 3, 192, 192)).astype(np.float32)
)
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    model,
    x,
    str(a.output),
    input_names=["log_rgb"],
    output_names=["field"],
    opset_version=17,
    dynamo=False,
)
net = onnx.load(a.output)
onnx.helper.set_model_props(
    net,
    {
        "contract": "openrevelare.newton.log-residual.v1",
        "normalization": "per-tile-log-mean-div4-eps1e-5",
        "training": "local synthetic multiplicative interference; not paired real Newton rings",
    },
)
onnx.checker.check_model(net)
onnx.save(net, a.output)
session = ort.InferenceSession(str(a.output), providers=["CPUExecutionProvider"])
with torch.inference_mode():
    expected = model(x).numpy()
actual = session.run(None, {"log_rgb": x.numpy()})[0]
error = float(np.max(np.abs(expected - actual)))
if error > 2e-5:
    raise RuntimeError(f"ONNX reference mismatch: {error}")
manifest = {
    "model_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
    "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    "max_onnx_reference_error": error,
    "onnx_ir": net.ir_version,
    "opset": 17,
}
for name in ("training-manifest", "validation"):
    path = a.checkpoint.parent / (name + ".json")
    if path.exists():
        manifest[name] = json.loads(path.read_text())
history = manifest.get("validation", [])
if history:
    manifest["validation_best_step"] = min(
        history, key=lambda row: row["after_mse"] + 2 * row.get("clean_mse", 0)
    )["step"]
manifest["source_at_export"] = {
    name: hashlib.sha256(
        (pathlib.Path(__file__).parent / name).read_bytes()
    ).hexdigest()
    for name in ("model.py", "train.py")
}
a.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
print(
    json.dumps(
        {
            k: v
            for k, v in manifest.items()
            if k not in ("training-manifest", "validation")
        },
        indent=2,
    )
)
