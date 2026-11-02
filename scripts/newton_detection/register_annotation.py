"""Register a user-drawn coarse annotation for evaluation ONLY; never called by detection."""

import argparse, hashlib, json, pathlib
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--screenshot", type=pathlib.Path, required=True)
p.add_argument("--reference", type=pathlib.Path, required=True)
p.add_argument("--polygons", type=pathlib.Path, required=True)
p.add_argument("--output", type=pathlib.Path, required=True)
a = p.parse_args()
query = cv2.imread(str(a.screenshot), cv2.IMREAD_GRAYSCALE)
reference = cv2.imread(str(a.reference), cv2.IMREAD_GRAYSCALE)
if query is None or reference is None:
    raise ValueError("Could not read registration images")
sift = cv2.SIFT_create(nfeatures=5000)
kq, dq = sift.detectAndCompute(query, None)
kr, dr = sift.detectAndCompute(reference, None)
pairs = cv2.BFMatcher().knnMatch(dq, dr, k=2)
good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
if len(good) < 20:
    raise ValueError("Insufficient independent image matches")
src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
cv2.setRNGSeed(4281)
h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3)
if h is None or inliers.sum() < 20:
    raise ValueError("Unreliable screenshot registration")
errors = np.linalg.norm(cv2.perspectiveTransform(src, h) - dst, axis=2).ravel()
height, width = reference.shape
polygons = json.loads(a.polygons.read_text())


def project(points):
    result = cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), h).reshape(
        -1, 2
    )
    return (result / np.array([width, height])).tolist()


result = {
    "schema": 1,
    "kind": "coarse_positive_regions_not_pixel_ground_truth",
    "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
    "screenshot_sha256": hashlib.sha256(a.screenshot.read_bytes()).hexdigest(),
    "registration": {
        "good_matches": len(good),
        "inliers": int(inliers.sum()),
        "median_inlier_error_pixels": float(
            np.median(errors[inliers.ravel().astype(bool)])
        ),
        "homography": h.tolist(),
    },
    "picture_support": project(
        [
            [0, 0],
            [query.shape[1] - 1, 0],
            [query.shape[1] - 1, query.shape[0] - 1],
            [0, query.shape[0] - 1],
        ]
    ),
    "regions": {key: project(value) for key, value in polygons.items()},
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2))
preview = cv2.imread(str(a.reference))
for polygon in result["regions"].values():
    points = (np.array(polygon) * [width, height]).round().astype(np.int32)
    cv2.polylines(preview, [points], True, (0, 0, 255), 3)
cv2.imwrite(str(a.output.with_suffix(".jpg")), preview)
print(json.dumps(result["registration"], indent=2))
