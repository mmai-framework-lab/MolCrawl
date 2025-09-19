#!/usr/bin/env python3
"""
OMIM Evaluation Visualization Script
====================================

OMIM (Online Mendelian Inheritance in Man) 評価結果の可視化スクリプト

生成される可視化:
1. 混同行列 (Confusion Matrix)
2. 性能指標チャート (Performance Metrics)
3. 遺伝形式別分析 (Inheritance Pattern Analysis)
4. ROC・PR曲線 (ROC & PR Curves)
5. 予測スコア分布 (Prediction Score Distribution)
6. 包括的HTMLレポート (Comprehensive HTML Report)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def setup_logging() -> logging.Logger:
    """ログ設定をセットアップ"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class OMIMVisualizationGenerator:
    """OMIM評価結果可視化生成クラス"""
    
    def __init__(self, results_dir: str, logger: Optional[logging.Logger] = None):
        self.results_dir = results_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 出力ディレクトリ設定
        self.viz_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # データロード
        self.results = self.load_results()
        self.predictions_df = self.load_predictions()
        
        # カラーパレット設定
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#F39C12', 
            'success': '#27AE60',
            'danger': '#E74C3C',
            'warning': '#F1C40F',
            'info': '#17A2B8',
            'dark': '#343A40'
        }

    def load_results(self) -> Dict:
        """評価結果をロード"""
        results_file = os.path.join(self.results_dir, 'omim_evaluation_results.json')
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Results file not found: {results_file}")
            return {}

    def load_predictions(self) -> pd.DataFrame:
        """予測結果をロード"""
        pred_file = os.path.join(self.results_dir, 'omim_predictions.csv')
        try:
            return pd.read_csv(pred_file)
        except FileNotFoundError:
            self.logger.error(f"Predictions file not found: {pred_file}")
            return pd.DataFrame()

    def generate_confusion_matrix(self) -> str:
        """混同行列を生成"""
        self.logger.info("Generating confusion matrix plot")
        
        if self.predictions_df.empty:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 混同行列計算
        cm = confusion_matrix(self.predictions_df['true_label'], self.predictions_df['prediction'])
        
        # ヒートマップ作成
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Benign', 'Disease-causing'],
                   yticklabels=['Benign', 'Disease-causing'])
        
        ax.set_title('OMIM Hereditary Disease Prediction\nConfusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # 統計情報追加
        if self.results:
            accuracy = self.results.get('accuracy', 0)
            f1_score = self.results.get('f1_score', 0)
            ax.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}\nF1-Score: {f1_score:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存
        output_file = os.path.join(self.viz_dir, 'confusion_matrix.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {output_file}")
        return output_file

    def generate_performance_metrics(self) -> str:
        """性能指標チャートを生成"""
        self.logger.info("Generating performance metrics chart")
        
        if not self.results:
            return ""
        
        # メトリクス抽出
        metrics = {
            'Accuracy': self.results.get('accuracy', 0),
            'Precision': self.results.get('precision', 0),
            'Recall': self.results.get('recall', 0),
            'F1-Score': self.results.get('f1_score', 0),
            'ROC-AUC': self.results.get('roc_auc', 0),
            'PR-AUC': self.results.get('pr_auc', 0),
            'Sensitivity': self.results.get('sensitivity', 0),
            'Specificity': self.results.get('specificity', 0)
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 基本メトリクス棒グラフ
        basic_metrics = {k: v for k, v in metrics.items() if k not in ['ROC-AUC', 'PR-AUC']}
        bars1 = ax1.bar(basic_metrics.keys(), basic_metrics.values(), 
                       color=[self.colors['primary'], self.colors['secondary'], 
                             self.colors['success'], self.colors['danger'],
                             self.colors['warning'], self.colors['info']])
        
        ax1.set_title('OMIM Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # 値をバーの上に表示
        for bar, value in zip(bars1, basic_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # AUCメトリクス
        auc_metrics = {'ROC-AUC': metrics['ROC-AUC'], 'PR-AUC': metrics['PR-AUC']}
        bars2 = ax2.bar(auc_metrics.keys(), auc_metrics.values(),
                       color=[self.colors['info'], self.colors['dark']])
        
        ax2.set_title('Area Under Curve Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('AUC Score', fontsize=12)
        ax2.set_ylim(0, 1)
        
        # 値をバーの上に表示
        for bar, value in zip(bars2, auc_metrics.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_file = os.path.join(self.viz_dir, 'performance_metrics.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance metrics chart saved to {output_file}")
        return output_file

    def generate_inheritance_analysis(self) -> str:
        """遺伝形式別分析チャートを生成"""
        self.logger.info("Generating inheritance pattern analysis")
        
        if not self.results or 'inheritance_analysis' not in self.results:
            self.logger.warning("No inheritance analysis data found")
            return ""
        
        inheritance_data = self.results['inheritance_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        patterns = list(inheritance_data.keys())
        
        # F1スコア比較
        f1_scores = [inheritance_data[p]['f1_score'] for p in patterns]
        bars1 = ax1.bar(patterns, f1_scores, color=self.colors['primary'])
        ax1.set_title('F1-Score by Inheritance Pattern', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1-Score')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 精度比較
        accuracies = [inheritance_data[p]['accuracy'] for p in patterns]
        bars2 = ax2.bar(patterns, accuracies, color=self.colors['secondary'])
        ax2.set_title('Accuracy by Inheritance Pattern', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # サンプル数
        sample_counts = [inheritance_data[p]['sample_count'] for p in patterns]
        bars3 = ax3.bar(patterns, sample_counts, color=self.colors['success'])
        ax3.set_title('Sample Count by Inheritance Pattern', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sample Count')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, sample_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(value), ha='center', va='bottom')
        
        # ポジティブ率
        positive_rates = [inheritance_data[p]['positive_count'] / inheritance_data[p]['sample_count'] 
                         for p in patterns]
        bars4 = ax4.bar(patterns, positive_rates, color=self.colors['danger'])
        ax4.set_title('Disease-causing Rate by Inheritance Pattern', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Disease-causing Rate')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, positive_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('OMIM Inheritance Pattern Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        output_file = os.path.join(self.viz_dir, 'inheritance_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Inheritance analysis chart saved to {output_file}")
        return output_file

    def generate_roc_pr_curves(self) -> str:
        """ROC曲線とPR曲線を生成"""
        self.logger.info("Generating ROC and PR curves")
        
        if self.predictions_df.empty:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        y_true = self.predictions_df['true_label']
        y_scores = self.predictions_df['prediction_score']
        
        # ROC曲線
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = self.results.get('roc_auc', 0)
        
        ax1.plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color=self.colors['danger'], lw=2, linestyle='--', alpha=0.8)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # PR曲線
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = self.results.get('pr_auc', 0)
        
        ax2.plot(recall, precision, color=self.colors['secondary'], lw=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_file = os.path.join(self.viz_dir, 'roc_pr_curves.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"ROC and PR curves saved to {output_file}")
        return output_file

    def generate_score_distribution(self) -> str:
        """予測スコア分布を生成"""
        self.logger.info("Generating prediction score distribution")
        
        if self.predictions_df.empty:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # クラス別スコア分布
        disease_scores = self.predictions_df[self.predictions_df['true_label'] == 1]['prediction_score']
        benign_scores = self.predictions_df[self.predictions_df['true_label'] == 0]['prediction_score']
        
        ax1.hist(benign_scores, bins=30, alpha=0.7, label='Benign', 
                color=self.colors['success'], density=True)
        ax1.hist(disease_scores, bins=30, alpha=0.7, label='Disease-causing', 
                color=self.colors['danger'], density=True)
        
        ax1.set_xlabel('Prediction Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Prediction Score Distribution by Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 閾値表示
        threshold = self.results.get('optimal_threshold', 0)
        ax1.axvline(threshold, color='black', linestyle='--', alpha=0.8, 
                   label=f'Optimal Threshold: {threshold:.3f}')
        ax1.legend()
        
        # ボックスプロット
        data_for_box = [benign_scores, disease_scores]
        box_plot = ax2.boxplot(data_for_box, labels=['Benign', 'Disease-causing'],
                              patch_artist=True)
        
        # ボックスプロットの色設定
        colors = [self.colors['success'], self.colors['danger']]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Prediction Score')
        ax2.set_title('Score Distribution Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_file = os.path.join(self.viz_dir, 'score_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Score distribution plot saved to {output_file}")
        return output_file

    def generate_html_report(self) -> str:
        """包括的HTMLレポートを生成"""
        self.logger.info("Generating HTML report")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMIM Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-name {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .image-title {{
            padding: 15px;
            background: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }}
        .summary-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        .inheritance-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .inheritance-table th, .inheritance-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .inheritance-table th {{
            background: #3498db;
            color: white;
            font-weight: bold;
        }}
        .inheritance-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 OMIM Hereditary Disease Prediction<br>Evaluation Report</h1>
        
        <div class="summary-section">
            <h2 style="color: white; border-bottom: 3px solid white;">📊 Executive Summary</h2>
            <p>This report presents the evaluation results of genome sequence model performance on OMIM (Online Mendelian Inheritance in Man) hereditary disease prediction task. The model was evaluated on {self.results.get('total_samples', 0)} variants with {self.results.get('positive_samples', 0)} disease-causing and {self.results.get('negative_samples', 0)} benign variants.</p>
        </div>

        <h2>📈 Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{self.results.get('accuracy', 0):.3f}</div>
                <div class="metric-name">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.results.get('precision', 0):.3f}</div>
                <div class="metric-name">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.results.get('recall', 0):.3f}</div>
                <div class="metric-name">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.results.get('f1_score', 0):.3f}</div>
                <div class="metric-name">F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.results.get('roc_auc', 0):.3f}</div>
                <div class="metric-name">ROC-AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.results.get('pr_auc', 0):.3f}</div>
                <div class="metric-name">PR-AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.results.get('sensitivity', 0):.3f}</div>
                <div class="metric-name">Sensitivity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.results.get('specificity', 0):.3f}</div>
                <div class="metric-name">Specificity</div>
            </div>
        </div>
        """
        
        # 遺伝形式別分析テーブル
        if 'inheritance_analysis' in self.results:
            html_content += """
        <h2>🧬 Inheritance Pattern Analysis</h2>
        <table class="inheritance-table">
            <thead>
                <tr>
                    <th>Inheritance Pattern</th>
                    <th>Sample Count</th>
                    <th>Disease-causing</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
            </thead>
            <tbody>
            """
            
            for pattern, metrics in self.results['inheritance_analysis'].items():
                html_content += f"""
                <tr>
                    <td>{pattern.replace('_', ' ').title()}</td>
                    <td>{metrics['sample_count']}</td>
                    <td>{metrics['positive_count']}</td>
                    <td>{metrics['accuracy']:.3f}</td>
                    <td>{metrics['precision']:.3f}</td>
                    <td>{metrics['recall']:.3f}</td>
                    <td>{metrics['f1_score']:.3f}</td>
                </tr>
                """
            
            html_content += """
            </tbody>
        </table>
        """
        
        # 画像セクション
        html_content += """
        <h2>📊 Visualization Results</h2>
        <div class="image-grid">
            <div class="image-card">
                <div class="image-title">Confusion Matrix</div>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="image-card">
                <div class="image-title">Performance Metrics</div>
                <img src="performance_metrics.png" alt="Performance Metrics">
            </div>
            <div class="image-card">
                <div class="image-title">Inheritance Pattern Analysis</div>
                <img src="inheritance_analysis.png" alt="Inheritance Analysis">
            </div>
            <div class="image-card">
                <div class="image-title">ROC & PR Curves</div>
                <img src="roc_pr_curves.png" alt="ROC PR Curves">
            </div>
            <div class="image-card">
                <div class="image-title">Score Distribution</div>
                <img src="score_distribution.png" alt="Score Distribution">
            </div>
        </div>
        """
        
        # フッター
        html_content += f"""
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>OMIM Hereditary Disease Prediction Evaluation System</p>
        </div>
    </div>
</body>
</html>
        """
        
        # HTMLファイル保存
        html_file = os.path.join(self.viz_dir, 'omim_evaluation_report.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to {html_file}")
        return html_file

    def generate_all_visualizations(self):
        """すべての可視化を生成"""
        self.logger.info("Generating all OMIM evaluation visualizations")
        
        # 各可視化生成
        confusion_matrix_file = self.generate_confusion_matrix()
        metrics_file = self.generate_performance_metrics()
        inheritance_file = self.generate_inheritance_analysis()
        curves_file = self.generate_roc_pr_curves()
        distribution_file = self.generate_score_distribution()
        html_file = self.generate_html_report()
        
        # 生成されたファイルをログ出力
        generated_files = [f for f in [confusion_matrix_file, metrics_file, inheritance_file, 
                                     curves_file, distribution_file, html_file] if f]
        
        self.logger.info(f"All visualizations completed. Files saved to {self.viz_dir}")
        self.logger.info(f"Generated {len(generated_files)} visualization files")
        
        return generated_files

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='OMIM Evaluation Visualization Generator')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing OMIM evaluation results')
    
    args = parser.parse_args()
    
    try:
        logger = setup_logging()
        logger.info("Starting OMIM visualization generation")
        
        # 可視化生成
        viz_generator = OMIMVisualizationGenerator(args.results_dir, logger)
        generated_files = viz_generator.generate_all_visualizations()
        
        logger.info("=== Visualization Generation Completed ===")
        logger.info(f"Output directory: {viz_generator.viz_dir}")
        logger.info(f"Generated {len(generated_files)} visualization files")
        logger.info(f"HTML report: {os.path.join(viz_generator.viz_dir, 'omim_evaluation_report.html')}")
        
    except Exception as e:
        print(f"Error during visualization generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
