import os, json, pickle, datetime, pathlib

# Paths
base_dir = pathlib.Path(r'D:\gemma4\segmentation_lattuce-desease')
report_dir = base_dir / 'Report'
images_dir = report_dir / 'images'

# Gather image list
image_files = [f.name for f in images_dir.iterdir() if f.is_file()]

# Build report dict
report_dict = {
    'project_path': str(base_dir),
    'generated_images': image_files,
    'notebook_copy_path': str(base_dir / '01_data_transforms_copy.ipynb'),
    'executed_notebook_path': str(base_dir / '01_data_transforms_executed.ipynb'),
    'execution_timestamp': datetime.datetime.now().isoformat(),
    'references_file': str(report_dir / 'references.md')
}

# Serialize to binary .bin (pickle)
bin_path = report_dir / 'technical_report.bin'
with open(bin_path, 'wb') as f:
    pickle.dump(report_dict, f)
print(f'Binary report written to {bin_path}')

# Write references markdown (manually curated from web search)
references_md = """# References

1. **Self‑supervised learning for plant disease segmentation** – Recent studies highlight the shift toward reducing dependence on expert‑annotated labels and leveraging large unlabeled datasets. DOI/URL: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEFPhuGBd2qh-nZhGnTgHktLinM5x6JHA6trlitiYO1WZFpkesoEWODSiWGY7hK7vd88c7a4tsq55iCCNPt5dOprioTXv_ojiFkvuw2YNsMRfh29knCmp-H61f1LVy0591U-y8=
2. **Domain‑specific augmentations and synthetic data generation** – Utilising affine transformations, posterization, and diffusion models (Stable Diffusion with ControlNet) to create realistic diseased leaf images. DOI/URL: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHF1kFF8N11Rpw9HecG9c7ws9-CZdH40DzFPEjS5276mcSt2CHBl_qbCe3tT5jKC5tWzhU2Go-RWzlbouJE4DO87KWU0CABqhjjLQH4d3kJl7tFIzDT57un8Pva-1iKg9EvI6dTwO_Z1xZqipPCfnZ_kBPaWuzoEpDugOrh5C5aEPmJnjs-6z9YNvZBrQ7OfoYfjwsX
3. **Contrastive and masked image modeling frameworks** – Combining contrastive learning, MIM, and BYOL for robust feature extraction. DOI/URL: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGeFZrzXxIFjdvFDltDLwYqomzWD2GmGb3zEfdkSJI7oAUH55_LTsr_5O3MCGu_LJgjQ-3ylRy3MypFmTeDOi5c5XNw71nS93hNyRXTNOZuij4MmmNZg9sCezxTCWxW5g5REGsU4jmrn4j19qcgk5aryx55gAm6YfUQwUr4s7VO8vbQGAawwqYxW1YmUVr3WgLB_Yu0H1PEjhk0wX40hh6D7U2lAspc
4. **Two‑stage detection‑segmentation pipelines** – Using YOLOv8 for leaf detection followed by DeepLabv3+ for lesion segmentation. DOI/URL: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGIvKssg7ERcSV74xpeDgkvs1zvX_p2zPw1Gk8Ri7sYrfeS7L8vx-H5NiG1df-qZuddG9O4CHf329r_TRHfD3A8xD-MnUOglL937ZsplpOPrCYupeuXhp5KLStrmx9frY1r_5UZSi1qLg3Li_k=
5. **Uncertainty‑guided pseudo‑labeling** – Iteratively refine segmentation masks without manual annotation. DOI/URL: https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHKXwtlPAe7lWUdu-jYaiUT9-sQ2qQXshR58iKRSPNN5Ox3EBukZwq1iFaDUW6t81P8n5WVhYEcoecyk6Dyx-vQnwHC8H2-IblZD4iK7a76Y1qtYMCz6PBrpY97S9-g2PbSVLbrDdoJpC9diqTSmD0plWrWsKw0mQrfoTWuoRqFkoB8Z1TawClKmtFlwdQGJQD6tEG0OnKB4jY2F89Qp_l5LQvmar1-l07ZE3usYLGm_BZUNhd4AXGrQpUA9cK0csPEoOEbO3mD3Y9oibeIN6_yyw==
"""

ref_path = report_dir / 'references.md'
with open(ref_path, 'w', encoding='utf-8') as f:
    f.write(references_md)
print(f'References markdown written to {ref_path}')
