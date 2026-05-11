# Testing Various

This folder keeps verification utilities separate from the main training code so the repository root stays focused on source entrypoints.

## Layout

- `smoke_tests/`
  End-to-end and stage-wise sanity checks for environment, dataset wiring, transforms, dataloaders, model assembly, and the full DRSA-Net shape/gradient path.
- `hardware_checks/`
  Small targeted experiments for device-specific behavior such as AMP or CUDA checks.

## Suggested commands

Run from the repository root:

```powershell
python testing_various/smoke_tests/step00_env_check.py
python testing_various/smoke_tests/smoke_test_drsa.py
python testing_various/smoke_tests/step01_dataset.py
python testing_various/smoke_tests/step02_transforms.py
python testing_various/smoke_tests/step03_dataloader.py
python testing_various/smoke_tests/step04_model_assembly.py
python testing_various/hardware_checks/scratch_amp_test.py
```

Generated artifacts remain outside Git because `drsa_net_output/` and `mlruns/` are ignored.
