import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from tqdm import tqdm

def debug_dataset():
    config = LabConfig().resolve()
    df = pd.read_csv(config.manifests_dir / "multirepresentation_manifest.csv")
    
    for split in ["train", "val"]:
        print(f"Checking {split} split...")
        ds = MultiChannelLeafDataset(df, config, split=split)
        for i in tqdm(range(len(ds))):
            try:
                sample = ds[i]
                for k, v in sample.items():
                    if v is None:
                        print(f"\n[ERROR] Key '{k}' is None at index {i} in {split} split")
                        print(f"Row: {ds.manifest_df.iloc[i].to_dict()}")
            except Exception as e:
                print(f"\n[ERROR] Exception at index {i} in {split} split: {e}")
                print(f"Row: {ds.manifest_df.iloc[i].to_dict()}")

if __name__ == "__main__":
    debug_dataset()
