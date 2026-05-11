from pathlib import Path

from superpixel_feature_graph_pipeline import PipelineConfig, process_dataset, print_recommendations


def main() -> None:
    config = PipelineConfig(
        output_base="feature_extraction_output_train_cuda",
        splits=("train",),
        superpixel_algorithm="slic",
        num_segments=128,
        adaptive_segments=False,
        segments_per_megapixel=1000,
        compactness=6.0,
        slic_sigma=0.8,
        context_dilation=3,
        k_feature_neighbors=1,
        feature_distance_threshold=2.5,
        crf_iterations=2,
        use_deep_features=True,
        deep_feature_models=("resnet18",),
        deep_batch_size=64,
        deep_crop_size=224,
        deep_context_dilation=4,
        deep_pretrained=True,
        device="cuda",
        overwrite=False,
    )

    print("Running train-only CUDA-aware pipeline")
    print_recommendations(config)
    summary = process_dataset(config)
    summary_path = Path(config.output_base) / "processing_summary.csv"
    print("")
    print("Completed train-only pipeline")
    print(summary.head())
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
