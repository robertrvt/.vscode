"""Run three controlled AOI comparisons using the supplied sample images.

Extract this file and the aoi_test_images folder into MERCURY_AOI, beside
aoi_trace_compare.py. Keep the original capture filenames unchanged.

Run: python aoi_run_defect_tests.py
Press Q in each result window to continue to the next case.

These are artificial image changes for testing the comparison pipeline.
They do not establish real-defect detection accuracy or electrical continuity.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "aoi_captures" / "reference" / "reference_20261013_171420_793299_1920x1080.png"
ORIGINAL = ROOT / "aoi_captures" / "inspection" / "inspection_20261013_171422_990458_1920x1080.png"
ROI = (864, 318, 148, 123)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="Save results without opening windows.")
    args = parser.parse_args()
    comparison_script = ROOT / "aoi_trace_compare.py"
    cases = [
        ("01_unchanged", ORIGINAL, "VISUAL MATCH"),
        ("02_simulated_gap", ROOT / "aoi_test_images" / "inspection_simulated_gap.png", "REINSPECT"),
        ("03_simulated_bridge", ROOT / "aoi_test_images" / "inspection_simulated_bridge.png", "REINSPECT"),
    ]
    required = [comparison_script, REFERENCE, *(path for _, path, _ in cases)]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
        print("Extract the test package beside aoi_trace_compare.py, keeping your original captures in place.")
        return 1
    print("Each case uses the same reference, selected area and comparison settings.", flush=True)
    if not args.no_gui:
        print("Press Q in each result window to continue to the next test.", flush=True)
    for name, inspection, expected in cases:
        print(f"\n{name} | Expected: {expected}", flush=True)
        command = [
            sys.executable, str(comparison_script),
            "--reference", str(REFERENCE), "--inspection", str(inspection),
            "--roi", *map(str, ROI),
            "--threshold", "20", "--min-area", "20", "--tolerance", "1",
            "--output", str(ROOT / "aoi_results" / "controlled_tests" / name),
        ]
        if args.no_gui:
            command.append("--no-gui")
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"Stopped because {name} could not complete. Review the message above.")
            return result.returncode
    print("\nThree comparisons completed. Review each comparison.png and report.txt.")
    print(f"Results: {ROOT / 'aoi_results' / 'controlled_tests'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
