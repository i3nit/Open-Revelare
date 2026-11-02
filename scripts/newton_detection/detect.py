"""Annotation-free directional periodicity evidence; NOT calibrated Newton-ring probability."""

import argparse, hashlib, json, pathlib, struct, time
import numpy as np
from scipy import fft, ndimage
from PIL import Image


def read_rgbf(path):
    with pathlib.Path(path).open("rb") as f:
        w, h = struct.unpack("<ii", f.read(8))
        return np.frombuffer(f.read(), dtype="<f4").reshape(h, w, 3).copy()


def evidence(image, max_edge=1024):
    h, w = image.shape[:2]
    scale = min(1, max_edge / max(h, w))
    w2 = max(1, round(w * scale))
    h2 = max(1, round(h * scale))
    # Resize float planes, not an 8-bit rendered proxy. Preserve the saved view's coordinates.
    image = np.stack(
        [
            np.asarray(
                Image.fromarray(image[:, :, c]).resize((w2, h2), Image.Resampling.BOX)
            )
            for c in range(3)
        ],
        axis=2,
    )
    logs = np.log(np.maximum(image, 1e-5))
    planes = np.stack(
        [logs.mean(2), logs[:, :, 0] - logs[:, :, 1], logs[:, :, 2] - logs[:, :, 1]]
    )
    pad = 160
    padded = np.pad(planes, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    spectra = fft.fft2(padded, workers=4)
    fy = fft.fftfreq(padded.shape[1])[:, None]
    fx = fft.fftfreq(padded.shape[2])[None, :]
    best = np.zeros((h2, w2), np.float32)
    mono = best.copy()
    colour = best.copy()
    phase_at_best = best.copy()
    period_at_best = best.copy()
    direction_at_best = best.copy()
    phase_colour = best.copy()
    periods = [4, 5, 6.3, 8, 10, 12.6, 16, 20, 25, 32]
    angles = [-20, -10, 0, 10, 20]
    for period in periods:
        high = planes - ndimage.gaussian_filter(
            planes, (0, period * 0.35, period * 0.35)
        )
        local_energy = (
            ndimage.gaussian_filter(high * high, (0, period, period * 1.5)) + 1e-7
        )
        for angle in angles:
            theta = np.deg2rad(angle)
            along = fx * np.cos(theta) + fy * np.sin(theta)
            normal = -fx * np.sin(theta) + fy * np.cos(theta)
            kernel = np.exp(
                -2
                * np.pi**2
                * (
                    (along * period * 1.5) ** 2
                    + ((normal - 1 / period) * period * 0.9) ** 2
                )
            ).astype(np.float32)
            response = fft.ifft2(spectra * kernel, workers=4)[:, pad:-pad, pad:-pad]
            power = np.abs(response) ** 2
            lum = np.clip(2 * power[0] / local_energy[0], 0, 1) * np.clip(
                2 * np.sqrt(power[0]) / 0.025, 0, 1
            )
            chrom = np.clip(
                2 * (power[1] + power[2]) / (local_energy[1] + local_energy[2]), 0, 1
            ) * np.clip(2 * np.sqrt(power[1] + power[2]) / 0.025, 0, 1)
            phase = np.abs((response[1] * response[2].conj()).imag) / np.maximum(
                np.abs(response[1]) * np.abs(response[2]), 1e-8
            )
            phase_colour = np.maximum(phase_colour, chrom * phase)
            score = np.maximum(lum, chrom)
            update = score > best
            best[update] = score[update]
            period_at_best[update] = period
            direction_at_best[update] = angle
            phase_at_best[update] = phase[update]
            mono = np.maximum(mono, lum)
            colour = np.maximum(colour, chrom)
    return dict(
        phase_chromatic=phase_colour,
        periodicity=best,
        luminance=mono,
        chromatic=colour,
        phase_support=phase_at_best,
        period=period_at_best,
        direction=direction_at_best,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    for path in a.input:
        start = time.monotonic()
        maps = evidence(read_rgbf(path))
        output = (
            a.output
            if len(a.input) == 1 and a.output.suffix == ".npz"
            else a.output / (path.stem + ".periodicity.npz")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **maps)
        metadata = {
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "script_sha256_at_inference": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()
            ).hexdigest(),
            "annotation_input": False,
            "semantic_probability": False,
            "periods": [4, 5, 6.3, 8, 10, 12.6, 16, 20, 25, 32],
            "angles": [-20, -10, 0, 10, 20],
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        print(
            json.dumps(
                dict(
                    file=path.name,
                    seconds=time.monotonic() - start,
                    annotation_input=False,
                    evidence_not_semantic_probability=True,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
