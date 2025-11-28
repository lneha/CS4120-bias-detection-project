"""
Statistical Analysis for Bias Detection Project
CS4120 Natural Language Processing - Northeastern University
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency
from scipy import stats

class BiasStatisticalAnalysis:
    def __init__(self, annotations_dir='../data/annotations'):
        """Load all annotation files"""
        self.annotations_dir = annotations_dir
        self.df = self.load_all_annotations()
        
    def load_all_annotations(self):
        """Load all JSON annotation files into single DataFrame"""
        all_data = []
        
        for filename in os.listdir(self.annotations_dir):
            if filename.endswith('_annotations.json'):
                filepath = os.path.join(self.annotations_dir, filename)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for ann in data['annotations']:
                    all_data.append(ann)
        
        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} total annotations")
        return df
    
    def descriptive_statistics(self):
        """Calculate comprehensive descriptive statistics"""
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS")
        print("="*70)
        
        print("\n1. Overall Composite Bias Scores:")
        print(f"   Mean: {self.df['composite_bias_score'].mean():.4f}")
        print(f"   Std Dev: {self.df['composite_bias_score'].std():.4f}")
        
        print("\n2. By Model:")
        print(self.df.groupby('model')['composite_bias_score'].agg(['mean', 'std']).round(4))
        
        print("\n3. By Topic:")
        print(self.df.groupby('topic')['composite_bias_score'].agg(['mean', 'std']).round(4))
    
    def anova_model_comparison(self):
        """ANOVA: Do models have different bias scores?"""
        print("\n" + "="*70)
        print("ANOVA TEST: Model Comparison")
        print("="*70)
        
        claude = self.df[self.df['model'] == 'claude']['composite_bias_score'].dropna()
        gpt4 = self.df[self.df['model'] == 'gpt4']['composite_bias_score'].dropna()
        gemini = self.df[self.df['model'] == 'gemini']['composite_bias_score'].dropna()
        
        f_stat, p_value = f_oneway(claude, gpt4, gemini)
        
        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("Result: SIGNIFICANT difference between models")
        else:
            print("Result: No significant difference")
    
    def create_visualizations(self, output_dir='../results/figures'):
        """Generate visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        sns.set_style("whitegrid")
        
        # Figure 1: Bias by Model
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='model', y='composite_bias_score')
        plt.title('Composite Bias Scores by Model', fontweight='bold')
        plt.ylabel('Bias Score (1-5)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/1_bias_by_model.png', dpi=300)
        print("  Saved: 1_bias_by_model.png")
        plt.close()
        
        # Figure 2: Heatmap
        plt.figure(figsize=(10, 6))
        pivot = self.df.pivot_table(
            values='composite_bias_score',
            index='topic',
            columns='model',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Bias Score Heatmap', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/2_bias_heatmap.png', dpi=300)
        print("  Saved: 2_bias_heatmap.png")
        plt.close()
    
    def export_dataset(self, output_dir='../data/analysis'):
        """Export complete dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f'{output_dir}/complete_dataset.csv'
        self.df.to_csv(output_file, index=False)
        print(f"\nExported: {output_file}")
    
    def run_complete_analysis(self):
        """Execute all analyses"""
        print("\n" + "="*70)
        print("COMPLETE STATISTICAL ANALYSIS")
        print("="*70)
        
        self.descriptive_statistics()
        self.anova_model_comparison()
        self.create_visualizations()
        self.export_dataset()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

def main():
    analyzer = BiasStatisticalAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
