# -*- coding: utf-8 -*-
"""
Visualize Five-Factor Scores Over Time
---------------------------------------
Create time series plots for the five FOMC policy factors.

Usage:
    python src/visualize_five_factors.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import warnings

# Configure matplotlib to handle fonts properly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

def load_five_factor_data():
    """Load five-factor analysis results."""
    csv_path = Path("output/five_factor_scores.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Five-factor results not found: {csv_path}\n"
                              "Please run: python src/five_factor_analysis.py")
    
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def create_time_series_plot(df):
    """Create time series plot for all five factors."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Five-Factor Analysis: FOMC Policy Dimensions Over Time', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    factors = [
        ('rate_score', 'Rate/Tightening (Short-term Path)', 'tab:red'),
        ('inf_score', 'Inflation Upward Pressure', 'tab:orange'),
        ('guidance_score', 'Forward Guidance/Path Language', 'tab:blue'),
        ('qt_score', 'Balance Sheet/QT (Term Premium)', 'tab:green'),
        ('growth_soft_score', 'Growth/Labor Market Softening', 'tab:purple'),
    ]
    
    for idx, (col, title, color) in enumerate(factors):
        ax = axes[idx // 2, idx % 2]
        
        # Plot line
        ax.plot(df['Date'], df[col], 'o-', color=color, linewidth=2, 
                markersize=4, alpha=0.7, label=title)
        
        # Add mean line
        mean_val = df[col].mean()
        ax.axhline(y=mean_val, color='gray', linestyle='--', linewidth=1, 
                  alpha=0.5, label=f'Mean: {mean_val:.3f}')
        
        # Highlight peak periods
        peak_value = df[col].max()
        peak_dates = df[df[col] >= df[col].quantile(0.90)]['Date']
        ax.scatter(peak_dates, df[df[col] >= df[col].quantile(0.90)][col],
                  color='red', s=80, alpha=0.6, zorder=5, label='Top 10%')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Score', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # Remove empty subplot
    fig.delaxes(axes[2, 1])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("output/five_factor_timeseries.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Time series plot saved to: {output_path}")
    plt.close()

def create_correlation_heatmap(df):
    """Create correlation heatmap for the five factors."""
    factor_cols = ['rate_score', 'inf_score', 'guidance_score', 'qt_score', 'growth_soft_score']
    factor_labels = ['Rate/\nTightening', 'Inflation\nPressure', 'Forward\nGuidance', 
                    'Balance\nSheet/QT', 'Growth/Labor\nSoftening']
    
    # Compute correlation
    corr = df[factor_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(corr, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(factor_labels)))
    ax.set_yticks(np.arange(len(factor_labels)))
    ax.set_xticklabels(factor_labels, fontsize=10)
    ax.set_yticklabels(factor_labels, fontsize=10)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add correlation values
    for i in range(len(factor_labels)):
        for j in range(len(factor_labels)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=11, 
                          fontweight='bold')
    
    ax.set_title('Factor Correlation Matrix\n(Higher correlation = factors move together)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("output/five_factor_correlation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved to: {output_path}")
    plt.close()

def create_radar_chart(df):
    """Create radar chart for recent vs historical average."""
    factor_cols = ['rate_score', 'inf_score', 'guidance_score', 'qt_score', 'growth_soft_score']
    factor_labels = ['Rate/\nTightening', 'Inflation\nPressure', 'Forward\nGuidance', 
                    'Balance\nSheet/QT', 'Growth/Labor\nSoftening']
    
    # Get recent (2024-2025) vs historical (2020-2023) averages
    df['Year'] = df['Date'].dt.year
    recent_avg = df[df['Year'] >= 2024][factor_cols].mean()
    historical_avg = df[df['Year'] < 2024][factor_cols].mean()
    
    # Normalize to 0-1 scale for better visualization
    max_vals = df[factor_cols].max()
    recent_norm = (recent_avg / max_vals).values
    historical_norm = (historical_avg / max_vals).values
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(factor_labels), endpoint=False).tolist()
    recent_norm = np.concatenate((recent_norm, [recent_norm[0]]))
    historical_norm = np.concatenate((historical_norm, [historical_norm[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, recent_norm, 'o-', linewidth=2, label='Recent (2024-2025)', 
            color='tab:red', markersize=8)
    ax.fill(angles, recent_norm, alpha=0.25, color='tab:red')
    
    ax.plot(angles, historical_norm, 's-', linewidth=2, label='Historical (2020-2023)', 
            color='tab:blue', markersize=8)
    ax.fill(angles, historical_norm, alpha=0.25, color='tab:blue')
    
    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(factor_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Policy Stance Comparison: Recent vs Historical\n(Normalized to max values)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("output/five_factor_radar.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Radar chart saved to: {output_path}")
    plt.close()

def print_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("FIVE-FACTOR ANALYSIS SUMMARY")
    print("=" * 80)
    
    factor_cols = ['rate_score', 'inf_score', 'guidance_score', 'qt_score', 'growth_soft_score']
    factor_labels = ['Rate/Tightening', 'Inflation Pressure', 'Forward Guidance', 
                    'Balance Sheet/QT', 'Growth/Labor Softening']
    
    print("\nPeriod Analysis:")
    print("-" * 80)
    df['Year'] = df['Date'].dt.year
    
    for year in sorted(df['Year'].unique()):
        year_df = df[df['Year'] == year]
        print(f"\n{year} (n={len(year_df)}):")
        for col, label in zip(factor_cols, factor_labels):
            avg = year_df[col].mean()
            print(f"  {label:25s}: {avg:.4f}")
    
    print("\n" + "=" * 80)

def main():
    """Main entry point."""
    print("=" * 80)
    print("Visualizing Five-Factor Analysis Results")
    print("=" * 80)
    
    # Load data
    df = load_five_factor_data()
    print(f"✓ Loaded {len(df)} documents from 2020 to 2025")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_time_series_plot(df)
    create_correlation_heatmap(df)
    create_radar_chart(df)
    
    # Print summary
    print_summary_stats(df)
    
    print("\n" + "=" * 80)
    print("✓ All visualizations complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - output/five_factor_timeseries.png")
    print("  - output/five_factor_correlation.png")
    print("  - output/five_factor_radar.png")

if __name__ == "__main__":
    main()
