#!/usr/bin/env python3
"""Require a fully passing public test run with no hidden skips."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


EXPECTED_SKIPS: set[str] = {
    "tests.test_artifact_identity_ledger::test_distributed_training_config_digest_matches_bytes",
    "tests.test_media_provenance_bindings::test_n25_transfer_renderer_reproduces_its_media_and_manifest",
    # Phase-local: the arrays are release assets, not in the tree, and the recorded replay
    # takes several minutes on CPU; both run with PINN_PHASE_ASSET_ROOT / PINN_PHASE_SLOW_TESTS=1.
    "tests.test_phaselocal_media::test_population_release_asset_identities_skip_cleanly_when_absent",
    "tests.test_phaselocal_media::test_release_assets_rerender_the_still_pixel_for_pixel",
    "tests.test_phaselocal_model::test_short_rollout_reproduces_the_recorded_96_output",
}

# Windows only: these rejection-path tests need symlink creation privilege, which a
# Windows account may lack. They may skip there; everywhere else they must run.
WINDOWS_OPTIONAL_SKIPS: set[str] = {
    "tests.test_n16_96_transfer::test_asset_verifier_refuses_extra_entries_only_inside_the_documented_hierarchy",
    "tests.test_public_tree_scan::test_release_scan_detects_directory_symlinks_and_generated_trees",
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
    optional = WINDOWS_OPTIONAL_SKIPS if os.name == "nt" else set()
    if not EXPECTED_SKIPS <= skipped <= EXPECTED_SKIPS | optional:
        print(f"test outcome refused: expected skips {sorted(EXPECTED_SKIPS)}"
              f" (optional here: {sorted(optional)}), got {sorted(skipped)}")
        return 1
    print(f"test outcome accepted: {len(cases) - len([case for case in cases if case.find('skipped') is not None])} passed, "
          f"{len([case for case in cases if case.find('skipped') is not None])} skipped, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
