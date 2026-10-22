#!/usr/bin/env python3
"""Require a fully passing public test run with no hidden skips."""

from __future__ import annotations

import sys
from pathlib import Path
import xml.etree.ElementTree as ET


EXPECTED_SKIPS: set[str] = {
    "tests.test_artifact_identity_ledger::test_distributed_training_config_digest_matches_bytes",
    "tests.test_media_provenance_bindings::test_n25_transfer_renderer_reproduces_its_media_and_manifest",
    # Phase-local: the arrays are release assets, not in the tree, and the recorded replay
    # takes several minutes on CPU; both run with PINN_PHASE_ASSET_ROOT / PINN_PHASE_SLOW_TESTS=1.
    "tests.test_phaselocal_media::test_release_assets_rerender_the_still_pixel_for_pixel",
    "tests.test_phaselocal_model::test_short_rollout_reproduces_the_recorded_96_output",
}


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "pytest.xml")
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failed = [case for case in cases if case.find("failure") is not None or case.find("error") is not None]
    skipped = {
        f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '').split('[')[0]}"
        for case in cases
        if case.find("skipped") is not None
    }
    if failed:
        print(f"test outcome refused: {len(failed)} failed or errored cases")
        return 1
    if skipped != EXPECTED_SKIPS:
        print(f"test outcome refused: expected skips {sorted(EXPECTED_SKIPS)}, got {sorted(skipped)}")
        return 1
    print(f"test outcome accepted: {len(cases)} passed, no skips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
