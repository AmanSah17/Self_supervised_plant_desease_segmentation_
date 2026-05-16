"""
Dashboard Report Generator
Creates comprehensive visualizations and analytics reports from inference results.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


class DashboardReportGenerator:
    """Generates comprehensive analytics dashboard reports."""
    
    def __init__(self, class_names: List[str], output_dir: Path):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_map = self._create_color_map()
        
    def _create_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        """Create RGB color map for disease classes."""
        colors = {
            0: (0, 0, 0),          # background (black)
            1: (0, 255, 0),        # BACT (green)
            2: (255, 0, 0),        # DML (red)
            3: (0, 0, 255),        # HLTY (blue - healthy)
            4: (255, 255, 0),      # PML (yellow)
            5: (255, 165, 0),      # SBL (orange)
            6: (128, 0, 128),      # SPW (purple)
            7: (0, 255, 255),      # VIRL (cyan)
            8: (255, 192, 203),    # WLBL (pink)
            9: (128, 128, 128),    # other (gray)
        }
        return colors
    
    def create_segmentation_visualization(self, 
                                         original_image: np.ndarray,
                                         predicted_mask: np.ndarray,
                                         image_stem: str) -> Path:
        """Create side-by-side visualization of original and segmented image."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original Image: {image_stem}", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Segmentation mask with colors
        segmented = self._apply_color_map(predicted_mask)
        axes[1].imshow(segmented)
        axes[1].set_title("Predicted Segmentation Mask", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Add legend
        legend_elements = [mpatches.Patch(facecolor=np.array(self.color_map[i])/255, 
                                         label=self.class_names[i-1] if i > 0 else 'Background')
                          for i in range(len(self.class_names) + 1)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)
        
        output_path = self.output_dir / f"{image_stem}_segmentation.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _apply_color_map(self, mask: np.ndarray) -> np.ndarray:
        """Apply RGB color mapping to segmentation mask."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.color_map.items():
            colored[mask == class_id] = color
        
        return colored
    
    def create_disease_spread_chart(self, results: List[Dict]) -> Path:
        """Create disease spread distribution charts."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        spreads = [r['disease_spread']['spread_score'] for r in results]
        confidences = [r['confidence_score'] for r in results]
        class_distributions = [r['disease_spread']['per_class_distribution'] for r in results]
        
        # 1. Disease spread histogram
        axes[0, 0].hist(spreads, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Disease Spread Score (%)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of Disease Spread Scores', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence score histogram
        axes[0, 1].hist(confidences, bins=15, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Confidence Score', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distribution of Model Confidence Scores', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot: spread vs confidence
        axes[1, 0].scatter(spreads, confidences, alpha=0.6, s=50, color='green')
        axes[1, 0].set_xlabel('Disease Spread Score (%)', fontsize=11)
        axes[1, 0].set_ylabel('Confidence Score', fontsize=11)
        axes[1, 0].set_title('Disease Spread vs Model Confidence', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistics box
        axes[1, 1].axis('off')
        stats_text = f"""
        DISEASE SPREAD STATISTICS
        
        Mean Spread Score: {np.mean(spreads):.2f}%
        Median Spread Score: {np.median(spreads):.2f}%
        Std Dev: {np.std(spreads):.2f}%
        Min: {np.min(spreads):.2f}%
        Max: {np.max(spreads):.2f}%
        
        Mean Confidence: {np.mean(confidences):.4f}
        Median Confidence: {np.median(confidences):.4f}
        Std Dev: {np.std(confidences):.4f}
        
        Total Samples: {len(results)}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        output_path = self.output_dir / "disease_spread_analysis.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_class_distribution_chart(self, results: List[Dict]) -> Path:
        """Create per-class distribution charts."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Aggregate class distribution
        class_totals = {class_name: 0 for class_name in self.class_names}
        
        for result in results:
            per_class = result['disease_spread']['per_class_distribution']
            for class_name, data in per_class.items():
                if class_name in class_totals:
                    class_totals[class_name] += data['pixel_count']
        
        # Bar chart
        classes = list(class_totals.keys())
        counts = list(class_totals.values())
        colors_list = [np.array(self.color_map[i+1])/255 for i in range(len(classes))]
        
        axes[0].barh(classes, counts, color=colors_list, edgecolor='black')
        axes[0].set_xlabel('Total Pixels Detected', fontsize=11)
        axes[0].set_title('Per-Class Pixel Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Pie chart
        total = sum(counts)
        percentages = [c/total*100 for c in counts]
        axes[1].pie(percentages, labels=classes, autopct='%1.1f%%', colors=colors_list,
                   startangle=90, textprops={'fontsize': 10})
        axes[1].set_title('Class Distribution Percentage', fontsize=12, fontweight='bold')
        
        output_path = self.output_dir / "class_distribution.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_segmentation_metrics_report(self, metrics: Dict) -> Path:
        """Create segmentation metrics visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        class_iou = metrics.get('class_iou', [])
        class_precision = metrics.get('class_precision', [])
        class_recall = metrics.get('class_recall', [])
        
        # Prepare class labels
        class_labels = [self.class_names[i-1] if i > 0 else 'Background' 
                       for i in range(len(class_iou))]
        
        # IoU chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_labels)))
        axes[0].barh(class_labels, class_iou, color=colors, edgecolor='black')
        axes[0].set_xlabel('IoU Score', fontsize=11)
        axes[0].set_title('Intersection over Union (IoU) per Class', fontsize=12, fontweight='bold')
        axes[0].set_xlim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Precision chart
        axes[1].barh(class_labels, class_precision, color=colors, edgecolor='black')
        axes[1].set_xlabel('Precision Score', fontsize=11)
        axes[1].set_title('Precision per Class', fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Recall chart
        axes[2].barh(class_labels, class_recall, color=colors, edgecolor='black')
        axes[2].set_xlabel('Recall Score', fontsize=11)
        axes[2].set_title('Recall per Class', fontsize=12, fontweight='bold')
        axes[2].set_xlim([0, 1])
        axes[2].grid(True, alpha=0.3, axis='x')
        
        output_path = self.output_dir / "segmentation_metrics.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_html_dashboard(self, results: List[Dict], metrics: Dict, 
                             output_filename: str = "dashboard.html") -> Path:
        """Create interactive HTML dashboard report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lettuce Disease Segmentation Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .dashboard {{
            padding: 30px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .stats-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .stat-item {{
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #eee;
        }}
        
        .stat-label {{
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        
        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }}
        
        .disease-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
            margin: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌱 Lettuce Disease Segmentation</h1>
            <p>Advanced Multi-Class Disease Detection Dashboard</p>
        </header>
        
        <div class="dashboard">
            <!-- Key Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Samples</div>
                    <div class="metric-value">{len(results)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean IoU</div>
                    <div class="metric-value">{metrics.get('miou', 0):.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Precision</div>
                    <div class="metric-value">{metrics.get('precision', 0):.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Recall</div>
                    <div class="metric-value">{metrics.get('recall', 0):.4f}</div>
                </div>
            </div>
            
            <!-- Disease Spread Analysis -->
            <div class="section">
                <h2>📊 Disease Spread Analysis</h2>
                <div class="chart-container">
                    <img src="disease_spread_analysis.png" alt="Disease Spread Analysis">
                </div>
            </div>
            
            <!-- Class Distribution -->
            <div class="section">
                <h2>🎯 Class Distribution</h2>
                <div class="chart-container">
                    <img src="class_distribution.png" alt="Class Distribution">
                </div>
            </div>
            
            <!-- Segmentation Metrics -->
            <div class="section">
                <h2>📈 Segmentation Metrics</h2>
                <div class="chart-container">
                    <img src="segmentation_metrics.png" alt="Segmentation Metrics">
                </div>
            </div>
            
            <!-- Disease Spread Summary Statistics -->
            <div class="section">
                <h2>📋 Disease Spread Statistics</h2>
                <div class="stats-box">
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Mean Spread Score</div>
                            <div class="stat-value">{np.mean([r['disease_spread']['spread_score'] for r in results]):.2f}%</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Median Spread Score</div>
                            <div class="stat-value">{np.median([r['disease_spread']['spread_score'] for r in results]):.2f}%</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Min Spread Score</div>
                            <div class="stat-value">{np.min([r['disease_spread']['spread_score'] for r in results]):.2f}%</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Max Spread Score</div>
                            <div class="stat-value">{np.max([r['disease_spread']['spread_score'] for r in results]):.2f}%</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Top Detections -->
            <div class="section">
                <h2>🏆 Top Detected Diseases</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Image</th>
                            <th>Dominant Disease</th>
                            <th>Spread Score</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f"""
                        <tr>
                            <td>{r['image_stem']}</td>
                            <td><span class="disease-badge">{r['dominant_disease_class']}</span></td>
                            <td>{r['disease_spread']['spread_score']:.2f}%</td>
                            <td>{r['confidence_score']:.4f}</td>
                        </tr>
                        """ for r in sorted(results, key=lambda x: x['disease_spread']['spread_score'], reverse=True)[:10]])}
                    </tbody>
                </table>
            </div>
        </div>
        
        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Lettuce Disease Segmentation System - Advanced Computer Vision Analysis</p>
        </footer>
    </div>
</body>
</html>
        """
        
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path


def generate_dashboard(inference_results_dir: Path):
    """Main function to generate complete dashboard reports."""
    # Load inference results
    results_file = inference_results_dir / "inference_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Inference results not found at {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    metrics = data['metrics']
    
    # Define class names
    class_names = ['BACT', 'DML', 'HLTY', 'PML', 'SBL', 'SPW', 'VIRL', 'WLBL']
    
    # Create dashboard generator
    output_dir = inference_results_dir / "dashboard_report"
    generator = DashboardReportGenerator(class_names, output_dir)
    
    print("[INFO] Generating disease spread analysis chart...")
    generator.create_disease_spread_chart(results)
    
    print("[INFO] Generating class distribution chart...")
    generator.create_class_distribution_chart(results)
    
    print("[INFO] Generating segmentation metrics chart...")
    generator.create_segmentation_metrics_report(metrics)
    
    print("[INFO] Generating HTML dashboard...")
    generator.create_html_dashboard(results, metrics)
    
    print(f"[SUCCESS] Dashboard reports saved to {output_dir}")
    print(f"Open {output_dir / 'dashboard.html'} to view the interactive dashboard")


if __name__ == "__main__":
    # Example usage - run from inference output directory
    from lettuce_ssl_segmentation_lab.config import LabConfig
    
    config = LabConfig()
    config.resolve()
    
    inference_dir = config.lab_root / "stage9_test_inference"
    generate_dashboard(inference_dir)
