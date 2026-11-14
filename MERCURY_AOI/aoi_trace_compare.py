"""MERCURY AOI - Step 2: Compare a selected area of PCB traces.

Run beside aoi_camera_test.py:
    python aoi_trace_compare.py

The newest color PNGs in aoi_captures/reference and aoi_captures/inspection
are selected automatically. The terminal prints their exact filenames.
Drag around the traces on the reference, then press Enter or Space.
Press C in the selection window to clear a selection, or Esc to cancel.


Requires the desktop opencv-python package (NumPy is included as a dependency).
    python -m pip install opencv-python

"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    raise SystemExit("Install OpenCV first: python -m pip install opencv-python") from None


ROOT = Path(__file__).resolve().parent


def read_image(path):
    """Read a color image, including from a Windows Unicode path."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_png(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Could not encode {path.name}")
    path.write_bytes(encoded.tobytes())


def newest_photo(folder):
    # Search only this folder, excluding its grayscale subfolder.
    candidates = list(folder.glob("*.png"))
    if not candidates:
        raise ValueError(f"No PNG photos in {folder}. Capture one first with R or S.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def align_images(reference, inspection):
    """Estimate one small global affine transform; never deform traces locally."""
    if reference.shape != inspection.shape:
        raise ValueError("Reference and inspection must have the same resolution. Retake a matching pair.")
    height, width = reference.shape[:2]
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ins_gray = cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY)
    if min(float(ref_gray.std()), float(ins_gray.std())) < 5:
        raise ValueError("Too little image detail for alignment. Include clear PCB features.")
    scale = min(960 / width, 540 / height, 1.0)
    small_size = (round(width * scale), round(height * scale))
    small_ref = cv2.resize(ref_gray, small_size).astype(np.float32) / 255
    small_ins = cv2.resize(ins_gray, small_size).astype(np.float32) / 255
    window = cv2.createHanningWindow(small_size, cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(small_ref.copy(), small_ins.copy(), window)
    warp = np.eye(2, 3, dtype=np.float32)
    if np.isfinite(shift).all():
        warp[:, 2] = shift
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(
            small_ref, small_ins, warp, cv2.MOTION_AFFINE, criteria, None, 5
        )
        # Account for independently rounded x/y resize factors.
        sx, sy = small_size[0] / width, small_size[1] / height
        warp[0, 1] *= sy / sx
        warp[1, 0] *= sx / sy
        warp[0, 2] /= sx
        warp[1, 2] /= sy
        score, warp = cv2.findTransformECC(
            ref_gray, ins_gray, warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-5), None, 5
        )
    except cv2.error as error:
        raise ValueError("Alignment did not converge. Use the same board view, distance and focus.") from error
    if not np.isfinite(warp).all() or not np.isfinite(score) or score < 0.90:
        raise ValueError("The photos do not align reliably. Retake the inspection without moving the setup.")
    scales = np.linalg.svd(warp[:, :2], compute_uv=False)
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], np.float32)
    movement = np.linalg.norm(corners @ warp.T - corners[:, :2], axis=1).max()
    if np.linalg.det(warp[:, :2]) <= 0 or scales.min() < 0.97 or scales.max() > 1.03 or movement > min(width, height) * 0.10:
        raise ValueError("Camera/board movement is too large for this fixed-fixture test. Retake a matching pair.")
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(inspection, warp, (width, height), flags=flags)
    valid = cv2.warpAffine(
        np.full((height, width), 255, np.uint8), warp, (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    )
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
    return aligned, valid, float(score), warp


def choose_roi(reference):
    """Let the operator select traces on a preview that fits a laptop screen."""
    height, width = reference.shape[:2]
    scale = min(1100 / width, 650 / height, 1.0)
    preview = cv2.resize(reference, (round(width * scale), round(height * scale)))
    title = "Select traces: drag rectangle, Enter confirms, Esc cancels"
    x, y, w, h = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return None
    sx, sy = preview.shape[1] / width, preview.shape[0] / height
    left, top = round(x / sx), round(y / sy)
    right = min(width, round((x + w) / sx))
    bottom = min(height, round((y + h) / sy))
    return left, top, right - left, bottom - top


def compare_area(reference, aligned, valid, roi, threshold=20, min_area=20, tolerance=1):
    """Compare local brightness ranges to tolerate a small amount of edge jitter.

    The tolerance is measured in ORIGINAL image pixels. Features at or below
    that scale may be ignored. No closing/opening is applied to connect traces.
    """
    x, y, w, h = roi
    height, width = reference.shape[:2]
    if min(x, y) < 0 or min(w, h) < 40 or x + w > width or y + h > height:
        raise ValueError("Choose an area at least 40 x 40 pixels, inside the reference image.")
    margin = tolerance + 2
    if min(w, h) <= 2 * margin:
        raise ValueError("The selected area is too small for this tolerance.")
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ins_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    if float(ref_gray[y:y+h, x:x+w].std()) < 5:
        raise ValueError("The selected area has too little detail. Include visible traces and their gaps.")
    ref_smooth = cv2.GaussianBlur(ref_gray, (3, 3), 0)
    ins_smooth = cv2.GaussianBlur(ins_gray, (3, 3), 0)
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    # Calculate the ranges on the full image before cropping to avoid crop-edge artifacts.
    valid_roi = cv2.erode(valid, kernel)[y:y+h, x:x+w] > 0
    valid_roi[:margin] = False
    valid_roi[-margin:] = False
    valid_roi[:, :margin] = False
    valid_roi[:, -margin:] = False
    interior_size = (w - 2 * margin) * (h - 2 * margin)
    if valid_roi.sum() < 0.98 * interior_size:
        raise ValueError("The inspection does not cover this selected area. Retake the same view.")
    ref_crop = ref_smooth[y:y+h, x:x+w]
    ins_crop = ins_smooth[y:y+h, x:x+w]
    offset = float(np.median(ref_crop[valid_roi].astype(np.float32) - ins_crop[valid_roi]))
    if abs(offset) > 15:
        raise ValueError("Brightness changed too much. Keep lighting/exposure fixed and retake the pair.")
    corrected = np.clip(ins_smooth.astype(np.float32) + offset, 0, 255).astype(np.uint8)
    residual = np.maximum.reduce([
        cv2.subtract(corrected, cv2.dilate(ref_smooth, kernel)),
        cv2.subtract(cv2.erode(ref_smooth, kernel), corrected),
        cv2.subtract(ref_smooth, cv2.dilate(corrected, kernel)),
        cv2.subtract(cv2.erode(corrected, kernel), ref_smooth),
    ])[y:y+h, x:x+w]
    residual[~valid_roi] = 0
    raw_mask = ((residual > threshold) & valid_roi).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    mask = np.zeros_like(raw_mask)
    regions = []
    for index in range(1, count):
        left, top, bw, bh, area = map(int, stats[index])
        if area >= min_area:
            mask[labels == index] = 255
            regions.append((left, top, bw, bh, area))
    direct = cv2.absdiff(ref_crop, corrected[y:y+h, x:x+w])
    average = float(direct[valid_roi].mean())
    reasons = []
    if regions:
        reasons.append(f"{len(regions)} changed region(s) exceed the comparison settings.")
    if average > 12:
        reasons.append("Large overall variation remains; check illumination, focus and alignment.")
    return {
        "status": "REINSPECT" if reasons else "VISUAL MATCH",
        "reasons": reasons,
        "reference_gray": ref_gray[y:y+h, x:x+w],
        "inspection_gray": ins_gray[y:y+h, x:x+w],
        "residual": residual, "mask": mask, "regions": regions,
        "brightness_offset": offset, "mean_abs_difference": average,
        "changed_percent": float(100 * np.count_nonzero(mask) / valid_roi.sum()),
    }


def make_overview(reference, aligned, roi, result):
    x, y, w, h = roi
    overlay = aligned.copy()
    patch = overlay[y:y+h, x:x+w]
    marked = result["mask"] > 0
    patch[marked] = (0.45 * patch[marked] + 0.55 * np.array([0, 220, 255])).astype(np.uint8)
    for left, top, bw, bh, _ in result["regions"]:
        cv2.rectangle(overlay, (x+left-3, y+top-3), (x+left+bw+3, y+top+bh+3), (0, 100, 255), 2)
    crop_overlay = overlay[y:y+h, x:x+w].copy()
    cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (255, 180, 0), 2)

    def panel(image, label):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tile = np.full((360, 550, 3), 28, np.uint8)
        scale = min(530 / image.shape[1], 310 / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
        th, tw = resized.shape[:2]
        yy, xx = 42 + (310-th)//2, (550-tw)//2
        tile[yy:yy+th, xx:xx+tw] = resized
        cv2.putText(tile, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
        return tile

    heat = cv2.applyColorMap(np.clip(result["residual"].astype(np.uint16)*6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    grid = np.vstack([
        np.hstack([panel(result["reference_gray"], "Reference - grayscale"), panel(result["inspection_gray"], "Inspection - aligned grayscale")]),
        np.hstack([panel(heat, "Difference strength (display amplified x6)"), panel(crop_overlay, "Changed areas highlighted for review")]),
    ])
    header = np.full((90, 1100, 3), 28, np.uint8)
    color = (0, 220, 255) if result["status"] == "REINSPECT" else (120, 220, 160)
    cv2.putText(header, f"MERCURY AOI | {result['status']}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(header, f"{len(result['regions'])} changed regions | Visual comparison only | Not an electrical continuity test", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([header, grid]), overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--inspection", type=Path)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--threshold", type=int, default=20, help="Brightness difference, 0-255; larger values are less sensitive.")
    parser.add_argument("--min-area", type=int, default=20, help="Minimum connected changed pixels; larger values ignore smaller changes.")
    parser.add_argument("--tolerance", type=int, default=1, help="Edge-position tolerance, 0-3 original image pixels; may hide very small defects.")
    parser.add_argument("--output", type=Path, default=ROOT / "aoi_results")
    parser.add_argument("--no-gui", action="store_true", help="Requires --roi; saves results without opening windows.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255 or args.min_area < 1 or not 0 <= args.tolerance <= 3:
        parser.error("Use threshold 0-255, min-area >= 1, and tolerance 0-3.")
    if args.no_gui and args.roi is None:
        parser.error("--no-gui requires --roi X Y WIDTH HEIGHT.")
    # Avoid excessive thread overhead when processing one pair at a time.
    cv2.setNumThreads(2)
    try:
        ref_path = (args.reference or newest_photo(ROOT / "aoi_captures" / "reference")).expanduser().resolve()
        ins_path = (args.inspection or newest_photo(ROOT / "aoi_captures" / "inspection")).expanduser().resolve()
        if ref_path == ins_path:
            raise ValueError("Choose two separate captures, not the same file twice.")
        print(f"Reference:  {ref_path}", flush=True)
        print(f"Inspection: {ins_path}", flush=True)
        reference, inspection = read_image(ref_path), read_image(ins_path)
        roi = tuple(args.roi) if args.roi else choose_roi(reference)
        if roi is None:
            print("Selection cancelled.")
            return 0
        print(f"Selected area (x, y, width, height): {roi}", flush=True)
        print("Aligning the images...", flush=True)
        aligned, valid, score, warp = align_images(reference, inspection)
        result = compare_area(reference, aligned, valid, roi, args.threshold, args.min_area, args.tolerance)
        overview, overlay = make_overview(reference, aligned, roi, result)
        output = args.output.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        for name, image in {
            "comparison.png": overview,
            "reference_gray.png": result["reference_gray"],
            "inspection_aligned_gray.png": result["inspection_gray"],
            "difference_strength.png": result["residual"],
            "candidate_mask.png": result["mask"],
            "highlighted_inspection.png": overlay,
        }.items():
            write_png(output / name, image)
        x, y, w, h = roi
        report = [
            "MERCURY AOI - Prototype grayscale comparison",
            f"Status: {result['status']}",
            f"Reference: {ref_path}", f"Inspection: {ins_path}",
            f"Image resolution: {reference.shape[1]} x {reference.shape[0]}",
            f"Area (x, y, width, height): {roi}",
            f"Alignment correlation: {score:.6f}",
            f"Alignment matrix (reference-to-inspection sampling): {warp.tolist()}",
            f"Threshold: {args.threshold}/255; minimum area: {args.min_area} pixels; tolerance: {args.tolerance} pixel(s)",
            f"Brightness correction: {result['brightness_offset']:+.1f}/255",
            f"Mean absolute grayscale difference after alignment/smoothing: {result['mean_abs_difference']:.3f}/255",
            f"Retained changed pixels: {result['changed_percent']:.4f}% of valid selected area",
            f"Changed regions: {len(result['regions'])}",
            *result["reasons"],
            "",
            "VISUAL MATCH is not production PASS and does not establish electrical continuity.",
            "REINSPECT means visible differences or excessive variation need operator review.",
            "Only the selected rectangle is evaluated. It is not an automatic trace segmentation.",
            "Defaults are preliminary: 3x3 smoothing, neighborhood tolerance and area filtering can hide small defects.",
            "Validate using multiple good captures and labeled defect examples before setting PASS/FAIL rules.",
            "",
            "Candidate regions: full-image x, y, width, height, changed pixel count",
            *[f"{x+left}, {y+top}, {bw}, {bh}, {area}" for left, top, bw, bh, area in result["regions"]],
        ]
        (output / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"\n{result['status']}: {len(result['regions'])} changed region(s).")
        print(f"Alignment correlation: {score:.6f}")
        print(f"Results saved to: {output}")
        print("Review comparison.png. No changes above these settings is not a validated PASS.")
        if not args.no_gui:
            title = "MERCURY AOI - Comparison (Q or Esc closes)"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1100, 810)
            cv2.imshow(title, overview)
            while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
                if cv2.waitKey(30) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        return 0
    except (OSError, ValueError, cv2.error) as error:
        print(f"REINSPECT / comparison stopped: {error}", file=sys.stderr)
        return 1
    finally:
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
