from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator


def main() -> None:
    config = LabConfig().resolve()
    orchestrator = SegmentationResearchOrchestrator(config)

    manifest_df, summary = orchestrator.build_manifest()
    orchestrator.write_research_outputs(summary)

    print("=" * 72)
    print("Lettuce SSL Segmentation Lab")
    print("=" * 72)
    print(f"Manifest rows          : {len(manifest_df)}")
    print(f"Healthy samples        : {summary.healthy_samples}")
    print(f"Diseased samples       : {summary.diseased_samples}")
    print(f"Felz raw coverage      : {summary.samples_with_felz_raw}")
    print(f"Felz boundary coverage : {summary.samples_with_felz_boundary}")
    print(f"Felz colored coverage  : {summary.samples_with_felz_colored}")
    print(f"Classes                : {', '.join(summary.classes_seen)}")
    print(f"Splits                 : {', '.join(summary.splits_seen)}")
    print(f"Logs dir               : {config.logs_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
