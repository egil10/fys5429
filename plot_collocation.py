import os
import pandas as pd
import matplotlib.pyplot as plt

bs_path = r"c:\Users\ofurn\Dokumenter\Github\fys5429\data\generated\bs_collocation.parquet"
hs_path = r"c:\Users\ofurn\Dokumenter\Github\fys5429\data\generated\hs_collocation.parquet"
out_dir = r"c:\Users\ofurn\Dokumenter\Github\fys5429\plots\eda"

os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = {'interior': 'blue', 'initial_condition': 'red', 'boundary_S_lower': 'green', 'boundary_S_upper': 'orange', 'boundary_v_lower': 'purple', 'boundary_v_upper': 'cyan'}

if os.path.exists(bs_path):
    df_bs = pd.read_parquet(bs_path)
    # Map any BS boundary names to standard colors if not exactly matching
    df_bs.loc[df_bs['point_type'].str.contains('boundary'), 'point_type'] = 'boundary (S limits)'
    colors_bs = {'interior': 'blue', 'initial_condition': 'red', 'boundary (S limits)': 'green'}
    for pt, df_sub in df_bs.groupby('point_type'):
        axes[0].scatter(df_sub['S'], df_sub['tau'], label=pt, color=colors_bs.get(pt, 'black'), s=2, alpha=0.5)
    axes[0].set_title("Black-Scholes (2D Domain)")
    axes[0].set_xlabel("S (Spot Price)")
    axes[0].set_ylabel("tau (Time to Maturity)")
    lgnd0 = axes[0].legend()
    for handle in lgnd0.legend_handles:
        handle.set_sizes([30.0])

if os.path.exists(hs_path):
    df_hs = pd.read_parquet(hs_path)
    # Group standard boundaries vs variance boundaries
    df_hs.loc[df_hs['point_type'].str.contains('boundary_S'), 'point_type'] = 'boundary (S limits)'
    df_hs.loc[df_hs['point_type'].str.contains('boundary_v'), 'point_type'] = 'boundary (v limits)'
    
    colors_hs = {'interior': 'blue', 'initial_condition': 'red', 'boundary (S limits)': 'green', 'boundary (v limits)': 'purple'}
    
    for pt, df_sub in df_hs.groupby('point_type'):
        axes[1].scatter(df_sub['S'], df_sub['tau'], label=pt, color=colors_hs.get(pt, 'black'), s=2, alpha=0.5)
    axes[1].set_title("Heston (S vs tau projection)")
    axes[1].set_xlabel("S (Spot Price)")
    axes[1].set_ylabel("tau (Time to Maturity)")
    lgnd1 = axes[1].legend()
    for handle in lgnd1.legend_handles:
        handle.set_sizes([30.0])

    for pt, df_sub in df_hs.groupby('point_type'):
        # Filter out purely initial condition points for v-projection to see interior vs bounds
        if pt == 'initial_condition': continue 
        axes[2].scatter(df_sub['S'], df_sub['v'], label=pt, color=colors_hs.get(pt, 'black'), s=2, alpha=0.5)
    axes[2].set_title("Heston (S vs v projection)")
    axes[2].set_xlabel("S (Spot Price)")
    axes[2].set_ylabel("v (Variance)")
    lgnd2 = axes[2].legend()
    for handle in lgnd2.legend_handles:
        handle.set_sizes([30.0])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "collocation_visuals.png"), dpi=150)
print(f"Plot saved to {os.path.join(out_dir, 'collocation_visuals.png')}")
