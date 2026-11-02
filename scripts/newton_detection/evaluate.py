"""Independent coarse-region localisation audit. Never feeds annotations into a detector."""

import argparse, hashlib, json, pathlib
import numpy as np
from PIL import Image, ImageDraw
from detect import read_rgbf


def polygon_mask(points, size):
    im = Image.new("1", size)
    ImageDraw.Draw(im).polygon([(x * size[0], y * size[1]) for x, y in points], fill=1)
    return np.asarray(im, dtype=bool)


def resize_map(array, size):
    return np.asarray(
        Image.fromarray(array.astype(np.float32)).resize(
            size, Image.Resampling.BILINEAR
        )
    )


def top_weights(score, support, fraction):
    values = score[support]
    count = len(values) * fraction
    threshold = np.partition(values, max(0, len(values) - int(np.ceil(count))))[
        max(0, len(values) - int(np.ceil(count)))
    ]
    higher = (score > threshold) & support
    ties = (score == threshold) & support
    result = higher.astype(np.float32)
    if ties.any():
        result[ties] = max(0, count - higher.sum()) / ties.sum()
    return result, float(threshold)


def audit(score, support, regions):
    union = np.logical_or.reduce(list(regions.values())) & support
    rows = []
    for fraction in (0.10, 0.15, 0.25):
        weights, threshold = top_weights(score, support, fraction)
        share = float(weights[union].sum() / max(weights.sum(), 1e-9))
        baseline = float(union.sum() / support.sum())
        rows.append(
            dict(
                budget=fraction,
                threshold=threshold,
                coarse_region_share=share,
                chance_region_share=baseline,
                enrichment=share / baseline,
                upper_selected_fraction=float(
                    weights[regions["upper"] & support].mean()
                ),
                lower_selected_fraction=float(
                    weights[regions["lower"] & support].mean()
                ),
            )
        )
    return dict(
        mean_inside=float(score[union].mean()),
        mean_outside=float(score[support & ~union].mean()),
        scores_are_not_calibrated_probabilities=True,
        nonzero_picture_fraction=float(np.mean(score[support] > 1e-6)),
        rank_budgets=rows,
    )


def overlay(base, weights, annotation, size):
    rgb = np.asarray(base.resize(size)).astype(np.float32)
    alpha = 0.65 * weights[:, :, None]
    out = Image.fromarray(
        np.clip(rgb * (1 - alpha) + np.array([255, 30, 20]) * alpha, 0, 255).astype(
            np.uint8
        )
    )
    draw = ImageDraw.Draw(out)
    for points in annotation["regions"].values():
        points = [(x * size[0], y * size[1]) for x, y in points]
        draw.line(points + [points[0]], fill=(40, 255, 100), width=2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=pathlib.Path, required=True)
    p.add_argument("--name", default="DSC01937")
    p.add_argument("--annotation", type=pathlib.Path, required=True)
    a = p.parse_args()
    annotation = json.loads(a.annotation.read_text())
    reference = a.directory / (a.name + ".jpg")
    if (
        hashlib.sha256(reference.read_bytes()).hexdigest()
        != annotation["reference_sha256"]
    ):
        raise ValueError("Annotation does not match this reference image")
    base = Image.open(reference).convert("RGB")
    width = 900
    height = round(base.height * width / base.width)
    size = (width, height)
    support = polygon_mask(annotation["picture_support"], size)
    regions = {
        key: polygon_mask(points, size) for key, points in annotation["regions"].items()
    }
    source_manifest = json.loads((a.directory / "sources.json").read_text())
    source_record = next(
        (row for row in source_manifest if pathlib.Path(row["File"]).stem == a.name),
        None,
    )
    if source_record is None:
        raise ValueError("Missing source manifest entry")
    methods = {}
    unavailable = []
    for name in ("legacy", "bands", "ai"):
        path = a.directory / (a.name + "." + name + ".rgbf")
        if not path.exists() or (
            name == "ai" and not source_record.get("AiModelSha256")
        ):
            unavailable.append(name)
            continue
        grid = read_rgbf(path)
        if grid.shape[:2] != (base.height, base.width):
            raise ValueError(
                "Detector map does not match reference dimensions: " + name
            )
        methods[name] = resize_map(grid[:, :, 0], size)
    path = a.directory / (a.name + ".periodicity.npz")
    if path.exists():
        metadata = json.loads(path.with_suffix(".json").read_text())
        if (
            metadata["input_sha256"]
            != hashlib.sha256(
                (a.directory / (a.name + ".rgbf")).read_bytes()
            ).hexdigest()
        ):
            raise ValueError("Periodicity map was inferred from a different source")
        bank = np.load(path)
        for key in ("periodicity", "chromatic", "phase_chromatic"):
            methods[key] = resize_map(bank[key], size)
    report = {
        "annotation_kind": annotation["kind"],
        "registration": annotation["registration"],
        "warning": "Coarse positive regions are not pixel labels. Region enrichment is NOT precision/recall or proof of Newton-ring recognition.",
        "unavailable_methods": unavailable,
        "methods": {},
    }
    thumb_size = (450, round(height * 0.5))
    mosaic = Image.new("RGB", (thumb_size[0] * 3, (thumb_size[1] + 30) * 2), "#14191f")
    draw = ImageDraw.Draw(mosaic)
    for index, (name, score) in enumerate(methods.items()):
        report["methods"][name] = audit(score, support, regions)
        weights, threshold = top_weights(score, support, 0.15)
        result = overlay(base, weights, annotation, size)
        result.save(a.directory / (a.name + "." + name + ".top15.jpg"))
        Image.fromarray(np.round(np.clip(score, 0, 1) * 255).astype(np.uint8)).save(
            a.directory / (a.name + "." + name + ".score.png")
        )
        if name == "phase_chromatic":
            for region, points in annotation["regions"].items():
                xs = [p[0] * width for p in points]
                ys = [p[1] * height for p in points]
                box = (
                    max(0, int(min(xs)) - 8),
                    max(0, int(min(ys)) - 8),
                    min(width, int(max(xs)) + 9),
                    min(height, int(max(ys)) + 9),
                )
                crop_width = box[2] - box[0]
                crop_height = box[3] - box[1]
                detail = Image.new("RGB", (crop_width * 2, crop_height))
                detail.paste(base.resize(size).crop(box), (0, 0))
                detail.paste(result.crop(box), (crop_width, 0))
                detail.save(
                    a.directory / (a.name + "." + region + ".detection-detail.jpg")
                )
        if index < 6:
            x = index % 3 * thumb_size[0]
            y = index // 3 * (thumb_size[1] + 30)
            mosaic.paste(result.resize(thumb_size), (x, y + 30))
            item = report["methods"][name]["rank_budgets"][1]
            draw.text(
                (x + 8, y + 8),
                f"{name}: top15 region enrichment {item['enrichment']:.2f}x",
                fill="white",
            )
    mosaic.save(a.directory / (a.name + ".detection-audit.jpg"))
    (a.directory / (a.name + ".metrics.json")).write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: value["rank_budgets"][1] for key, value in report["methods"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
