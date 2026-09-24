"""
Gera imagens de alta qualidade para o README.md e documentação técnica.
- docs/images/cluster_pca_scatter.png
- docs/images/cluster_profiles_comparison.png
- docs/images/elbow_silhouette.png
- docs/images/golden_set_scorecard.png
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("docs/images", exist_ok=True)

# Configurações visuais modernas e limpas
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Cores oficiais dos 4 clusters
CLUSTER_COLORS = {
    0: "#E67E22",  # Low Output / Weak Defense (Laranja suave)
    1: "#2980B9",  # Striker (Azul vibrante)
    2: "#27AE60",  # Wrestler / Grappler (Verde esmeralda)
    3: "#8E44AD",  # Submission Specialist (Grappler) (Roxo)
}

CLUSTER_NAMES = {
    0: "Low Output / Weak Defense",
    1: "Striker",
    2: "Wrestler / Grappler",
    3: "Submission Specialist",
}

# -------------------------------------------------------------
# 1. CLUSTER PCA SCATTER PLOT COM LUTADORES ICÔNICOS
# -------------------------------------------------------------
print("Gerando docs/images/cluster_pca_scatter.png...")
df_clustered = pd.read_parquet("data/processed/fighters_clustered_labeled.parquet")
df_features = pd.read_parquet("data/processed/fighters_features.parquet")
pca = joblib.load("models/pca.pkl")
kmeans = joblib.load("models/kmeans.pkl")

pc1 = df_features["pc_1"].values
pc2 = df_features["pc_2"].values

fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FAFAFA")
ax.grid(True, linestyle="--", alpha=0.5, color="#D5D8DC")

# Plota cada cluster com transparência
for c_id, name in CLUSTER_NAMES.items():
    mask = df_clustered["cluster"] == c_id
    ax.scatter(
        pc1[mask],
        pc2[mask],
        c=CLUSTER_COLORS[c_id],
        label=f"Cluster {c_id}: {name} ({mask.sum()} lutadores)",
        alpha=0.55,
        s=35,
        edgecolors="none",
    )

# Centroides em PCA
centers_pca = kmeans.cluster_centers_[:, :2]
ax.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    s=250,
    c="#111111",
    marker="X",
    edgecolors="white",
    linewidths=2,
    label="Centroides dos Clusters",
    zorder=10,
)

# Anotações de 4 lutadores icônicos bem distribuídos espacialmente (sem sobreposição)
highlight_fighters = [
    ("Alex Pereira", "Alex Pereira (Striker)", (-40, 60), "#1B4F72"),
    ("Khabib Nurmagomedov", "Khabib Nurmagomedov (Wrestler / Grappler)", (50, 45), "#145A32"),
    ("Charles Oliveira", "Charles Oliveira (Grappler)", (50, -35), "#196F3D"),
    ("Robert Drysdale", "Robert Drysdale (Submission Specialist)", (-120, 25), "#512E5F"),
]

for query_name, display_label, offset, box_color in highlight_fighters:
    match = df_features[df_features["name"].str.lower() == query_name.lower()]
    if not match.empty:
        idx = match.index[0]
        pos = df_features.index.get_loc(idx)
        x_val = pc1[pos]
        y_val = pc2[pos]
        ax.scatter([x_val], [y_val], color="#E74C3C", s=90, edgecolors="black", linewidths=1.5, zorder=15)
        ax.annotate(
            display_label,
            (x_val, y_val),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#1A252F",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=box_color, alpha=0.95, lw=1.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=box_color, lw=1.5),
            zorder=20,
        )

ax.set_xlim(-6, 39)
ax.set_ylim(-16, 14)


ax.set_title("UFC Fighters — Agrupamento por Estilo de Luta (Projeção PCA)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% da variância explicada)", fontsize=10, fontweight="semibold")
ax.set_ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% da variância explicada)", fontsize=10, fontweight="semibold")
ax.legend(frameon=True, facecolor="white", edgecolor="#D5D8DC", fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("docs/images/cluster_pca_scatter.png")
plt.close()

# -------------------------------------------------------------
# 2. COMPARATIVO DAS MÉTRICAS DE PERFORMANCE POR CLUSTER
# -------------------------------------------------------------
print("Gerando docs/images/cluster_profiles_comparison.png...")
df_profile = pd.read_csv("reports/cluster_profile.csv")

metrics_to_plot = [
    ("significant_strikes_landed_per_minute", "Golpes Conectados / min"),
    ("significant_strikes_absorbed_per_minute", "Golpes Absorvidos / min"),
    ("average_takedowns_landed_per_15_minutes", "Quedas Aplicadas / 15 min"),
    ("average_submissions_attempted_per_15_minutes", "Submissões / 15 min"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

for ax, (metric, label) in zip(axes.flatten(), metrics_to_plot):
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#D5D8DC")
    
    bars = ax.barh(
        [CLUSTER_NAMES[c] for c in df_profile["cluster"]],
        df_profile[metric],
        color=[CLUSTER_COLORS[c] for c in df_profile["cluster"]],
        edgecolor="#333333",
        linewidth=0.8,
        height=0.6,
    )
    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=9)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max(df_profile[metric]) * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlim(0, max(df_profile[metric]) * 1.22)

plt.suptitle("Métricas Médias Reais por Estilo de Luta (UFCStats.com)", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
plt.savefig("docs/images/cluster_profiles_comparison.png")
plt.close()

# -------------------------------------------------------------
# 3. SELEÇÃO DO NÚMERO ÓTIMO DE CLUSTERS (ELBOW & SILHOUETTE)
# -------------------------------------------------------------
print("Gerando docs/images/elbow_silhouette.png...")
from sklearn.cluster import KMeans

with open("models/model_choice.json", "r", encoding="utf-8") as f:
    choice_data = json.load(f)

k_range = [3, 4, 5, 6, 7, 8]
silhouettes = [choice_data["silhouette_by_k"][str(k)] for k in k_range]

pca_cols = [c for c in df_features.columns if c.startswith("pc_")]
X_matrix = df_features[pca_cols].values
inertias = [KMeans(n_clusters=k, random_state=42, n_init=1).fit(X_matrix).inertia_ for k in k_range]

fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
fig.patch.set_facecolor("#FFFFFF")
ax1.set_facecolor("#FAFAFA")
ax1.grid(True, linestyle="--", alpha=0.5, color="#D5D8DC")

color_in = "#C0392B"
ax1.plot(k_range, inertias, marker="o", linewidth=2.2, color=color_in, label="Inércia (Elbow)")
ax1.set_xlabel("Número de Clusters (k)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Inércia (Soma dos Quadrados Intra-Cluster)", color=color_in, fontsize=10, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color_in)

ax2 = ax1.twinx()
color_sil = "#2980B9"
ax2.plot(k_range, silhouettes, marker="s", linewidth=2.2, linestyle="--", color=color_sil, label="Silhouette Score")
ax2.set_ylabel("Silhouette Score Médio", color=color_sil, fontsize=10, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color_sil)

ax1.axvline(x=4, color="#27AE60", linestyle=":", linewidth=2, label="k=4 Escolhido")
ax2.scatter([4], [silhouettes[1]], color="#27AE60", s=130, zorder=10, edgecolors="black")

plt.title("Otimização de Hiperparâmetro: Método Elbow e Coeficiente Silhouette", fontsize=12, fontweight="bold", pad=12)
fig.tight_layout()
plt.savefig("docs/images/elbow_silhouette.png")
plt.close()

# -------------------------------------------------------------
# 4. SCORECARD VISUAL DO GOLDEN SET DE 12 LUTADORES
# -------------------------------------------------------------
print("Gerando docs/images/golden_set_scorecard.png...")
with open("reports/golden_set_validation.json", "r", encoding="utf-8") as f:
    fighters_list = json.load(f)

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
fig.patch.set_facecolor("#FFFFFF")
ax.axis("off")

table_data = []
row_colors = []
correct_count = 0
for f in fighters_list:
    name = f["name"]
    expected = f["expected"]
    predicted = f["predicted"]
    match = f["correct"]
    if match:
        correct_count += 1
    status = "OK (Correto)" if match else "Divergente (Diluição)"
    table_data.append([name, expected, predicted, status])
    row_colors.append("#D4EFDF" if match else "#FADBD8")

columns = ["Lutador Conhecido", "Estilo Consagrado", "Estilo Previsto (ML)", "Validação"]
table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.30, 0.23, 0.32, 0.15],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

for j in range(len(columns)):
    cell = table[(0, j)]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)

for i, color in enumerate(row_colors, start=1):
    for j in range(len(columns)):
        cell = table[(i, j)]
        cell.set_facecolor(color)
        if j == 3:
            cell.set_text_props(fontweight="bold")

accuracy = (correct_count / len(fighters_list)) * 100
plt.title(
    f"Validação Supervisionada Golden Set — Acurácia: {accuracy:.1f}% ({correct_count}/{len(fighters_list)} Acertos)",
    fontsize=13,
    fontweight="bold",
    pad=20,
)
plt.tight_layout()
plt.savefig("docs/images/golden_set_scorecard.png")
plt.close()

print("Todas as 4 imagens foram geradas com sucesso em docs/images/!")

