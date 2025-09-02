#!/usr/bin/env python3
"""
ClinVar評価結果可視化スクリプト

ClinVar評価の結果を可視化し、詳細な分析を行います。
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClinVarResultsVisualizer:
    """ClinVar評価結果の可視化クラス"""
    
    def __init__(self, results_file, output_dir='./visualization_results'):
        """
        初期化
        
        Args:
            results_file (str): 評価結果JSONファイルのパス
            output_dir (str): 可視化結果の出力ディレクトリ
        """
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果の読み込み
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # プロット設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self):
        """混同行列をプロット"""
        logger.info("Creating confusion matrix plot")
        
        cm = self.results['confusion_matrix']
        matrix = np.array([
            [cm['true_negative'], cm['false_positive']],
            [cm['false_negative'], cm['true_positive']]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Predicted Benign', 'Predicted Pathogenic'],
                   yticklabels=['Actual Benign', 'Actual Pathogenic'])
        
        plt.title('Confusion Matrix - ClinVar Pathogenicity Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 正確度を追加
        accuracy = self.results['accuracy']
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'confusion_matrix.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_performance_metrics(self):
        """性能指標の棒グラフをプロット"""
        logger.info("Creating performance metrics plot")
        
        metrics = {
            'Accuracy': self.results['accuracy'],
            'Precision': self.results['precision'],
            'Recall': self.results['recall'],
            'F1-Score': self.results['f1_score'],
            'Sensitivity': self.results['sensitivity'],
            'Specificity': self.results['specificity']
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        
        plt.title('Model Performance Metrics on ClinVar Data')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_metrics.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_auc_comparison(self):
        """AUC指標の比較プロット"""
        logger.info("Creating AUC comparison plot")
        
        auc_metrics = {
            'ROC-AUC': self.results['roc_auc'],
            'PR-AUC': self.results['pr_auc']
        }
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(auc_metrics.keys(), auc_metrics.values(),
                      color=['#ff7f0e', '#2ca02c'])
        
        plt.title('Area Under Curve (AUC) Metrics')
        plt.ylabel('AUC Score')
        plt.ylim(0, 1)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, auc_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # ランダム予測の基準線を追加
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Prediction')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'auc_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'auc_metrics.pdf', bbox_inches='tight')
        plt.close()
    
    def create_performance_radar_chart(self):
        """性能指標のレーダーチャートを作成"""
        logger.info("Creating performance radar chart")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity']
        values = [
            self.results['accuracy'],
            self.results['precision'],
            self.results['recall'],
            self.results['f1_score'],
            self.results['sensitivity'],
            self.results['specificity']
        ]
        
        # レーダーチャート用の角度計算
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 円を閉じるために最初の値を追加
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # データをプロット
        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')
        
        # ラベルを設定
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        
        # グリッドの設定
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.title('Model Performance Radar Chart', size=16, y=1.1)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_radar.pdf', bbox_inches='tight')
        plt.close()
    
    def create_classification_report_table(self):
        """分類レポートテーブルを作成"""
        logger.info("Creating classification report table")
        
        # 分類レポート用データを準備
        cm = self.results['confusion_matrix']
        
        # クラス別の統計を計算
        benign_precision = cm['true_negative'] / (cm['true_negative'] + cm['false_negative']) if (cm['true_negative'] + cm['false_negative']) > 0 else 0
        benign_recall = cm['true_negative'] / (cm['true_negative'] + cm['false_positive']) if (cm['true_negative'] + cm['false_positive']) > 0 else 0
        benign_f1 = 2 * (benign_precision * benign_recall) / (benign_precision + benign_recall) if (benign_precision + benign_recall) > 0 else 0
        
        pathogenic_precision = self.results['precision']
        pathogenic_recall = self.results['recall']
        pathogenic_f1 = self.results['f1_score']
        
        # テーブルデータを作成
        report_data = {
            'Class': ['Benign', 'Pathogenic', 'Macro Avg', 'Weighted Avg'],
            'Precision': [
                benign_precision,
                pathogenic_precision,
                (benign_precision + pathogenic_precision) / 2,
                (benign_precision * (cm['true_negative'] + cm['false_positive']) + 
                 pathogenic_precision * (cm['true_positive'] + cm['false_negative'])) / 
                (cm['true_negative'] + cm['false_positive'] + cm['true_positive'] + cm['false_negative'])
            ],
            'Recall': [
                benign_recall,
                pathogenic_recall,
                (benign_recall + pathogenic_recall) / 2,
                (benign_recall * (cm['true_negative'] + cm['false_positive']) + 
                 pathogenic_recall * (cm['true_positive'] + cm['false_negative'])) / 
                (cm['true_negative'] + cm['false_positive'] + cm['true_positive'] + cm['false_negative'])
            ],
            'F1-Score': [
                benign_f1,
                pathogenic_f1,
                (benign_f1 + pathogenic_f1) / 2,
                (benign_f1 * (cm['true_negative'] + cm['false_positive']) + 
                 pathogenic_f1 * (cm['true_positive'] + cm['false_negative'])) / 
                (cm['true_negative'] + cm['false_positive'] + cm['true_positive'] + cm['false_negative'])
            ],
            'Support': [
                cm['true_negative'] + cm['false_positive'],
                cm['true_positive'] + cm['false_negative'],
                cm['true_negative'] + cm['false_positive'] + cm['true_positive'] + cm['false_negative'],
                cm['true_negative'] + cm['false_positive'] + cm['true_positive'] + cm['false_negative']
            ]
        }
        
        df_report = pd.DataFrame(report_data)
        
        # テーブルをプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_report.round(3).values,
                        colLabels=df_report.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # ヘッダーのスタイル設定
        for i in range(len(df_report.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Classification Report', size=16, pad=20)
        plt.savefig(self.output_dir / 'classification_report.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'classification_report.pdf', bbox_inches='tight')
        plt.close()
        
        # CSVとしても保存
        df_report.to_csv(self.output_dir / 'classification_report.csv', index=False)
    
    def create_summary_dashboard(self):
        """全体的なサマリーダッシュボードを作成"""
        logger.info("Creating summary dashboard")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 2x2のグリッドを作成
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 混同行列
        ax1 = fig.add_subplot(gs[0, 0])
        cm = self.results['confusion_matrix']
        matrix = np.array([
            [cm['true_negative'], cm['false_positive']],
            [cm['false_negative'], cm['true_positive']]
        ])
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Pred Benign', 'Pred Pathogenic'],
                   yticklabels=['Act Benign', 'Act Pathogenic'],
                   ax=ax1)
        ax1.set_title('Confusion Matrix')
        
        # 2. 性能指標バーチャート
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [self.results['accuracy'], self.results['precision'], 
                 self.results['recall'], self.results['f1_score']]
        bars = ax2.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('Performance Metrics')
        ax2.set_ylim(0, 1)
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. AUC指標
        ax3 = fig.add_subplot(gs[1, 0])
        auc_metrics = ['ROC-AUC', 'PR-AUC']
        auc_values = [self.results['roc_auc'], self.results['pr_auc']]
        bars = ax3.bar(auc_metrics, auc_values, color=['#ff7f0e', '#2ca02c'])
        ax3.set_title('AUC Metrics')
        ax3.set_ylim(0, 1)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        for bar, value in zip(bars, auc_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. 主要統計情報
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        stats_text = f"""
        Model Performance Summary
        
        Overall Accuracy: {self.results['accuracy']:.3f}
        
        Sensitivity (Recall): {self.results['sensitivity']:.3f}
        Specificity: {self.results['specificity']:.3f}
        
        Positive Predictive Value: {self.results['precision']:.3f}
        
        F1-Score: {self.results['f1_score']:.3f}
        
        ROC-AUC: {self.results['roc_auc']:.3f}
        PR-AUC: {self.results['pr_auc']:.3f}
        
        Total Variants Evaluated: {sum(cm.values())}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('ClinVar Evaluation Dashboard - Genome Sequence Model', fontsize=16, y=0.98)
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'summary_dashboard.pdf', bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """全ての可視化を生成"""
        logger.info("Generating all visualizations")
        
        self.plot_confusion_matrix()
        self.plot_performance_metrics()
        self.plot_auc_comparison()
        self.create_performance_radar_chart()
        self.create_classification_report_table()
        self.create_summary_dashboard()
        
        logger.info(f"All visualizations saved to {self.output_dir}")
    
    def create_html_report(self):
        """HTML形式の総合レポートを作成"""
        logger.info("Creating HTML report")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ClinVar Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ClinVar Pathogenicity Prediction Evaluation</h1>
                <p>Genome Sequence Model Performance Report</p>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{self.results['accuracy']:.3f}</div>
                    <div>Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.results['precision']:.3f}</div>
                    <div>Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.results['recall']:.3f}</div>
                    <div>Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.results['f1_score']:.3f}</div>
                    <div>F1-Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.results['roc_auc']:.3f}</div>
                    <div>ROC-AUC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.results['pr_auc']:.3f}</div>
                    <div>PR-AUC</div>
                </div>
            </div>
            
            <h2>Summary Dashboard</h2>
            <div class="image-container">
                <img src="summary_dashboard.png" alt="Summary Dashboard">
            </div>
            
            <h2>Detailed Analysis</h2>
            
            <h3>Confusion Matrix</h3>
            <div class="image-container">
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            
            <h3>Performance Metrics</h3>
            <div class="image-container">
                <img src="performance_metrics.png" alt="Performance Metrics">
            </div>
            
            <h3>Classification Report</h3>
            <div class="image-container">
                <img src="classification_report.png" alt="Classification Report">
            </div>
            
            <h3>Performance Radar Chart</h3>
            <div class="image-container">
                <img src="performance_radar.png" alt="Performance Radar Chart">
            </div>
            
        </body>
        </html>
        """
        
        with open(self.output_dir / 'evaluation_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info("HTML report created: evaluation_report.html")

def main():
    parser = argparse.ArgumentParser(description='Visualize ClinVar evaluation results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to evaluation results JSON file')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='Output directory for visualizations')
    parser.add_argument('--html_report', action='store_true',
                       help='Generate HTML report')
    
    args = parser.parse_args()
    
    try:
        visualizer = ClinVarResultsVisualizer(args.results_file, args.output_dir)
        
        # 全ての可視化を生成
        visualizer.generate_all_visualizations()
        
        # HTMLレポートを生成
        if args.html_report:
            visualizer.create_html_report()
        
        logger.info("Visualization completed successfully")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise

if __name__ == "__main__":
    main()
