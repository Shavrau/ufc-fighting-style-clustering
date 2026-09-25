"""
src/generate_readme_images.py

Gera imagens em alta resolução (300 DPI) para o README.md e redes sociais (LinkedIn):
1. docs/images/cluster_pca_scatter.png (com Striker, Well-rounded [GSP], Wrestler, Sub. Specialist e Cluster 0 [CM Punk])
2. docs/images/cluster_profiles_comparison.png (métricas reais dos clusters)
3. docs/images/elbow_silhouette.png (curvas de otimização de k)
4. docs/images/golden_set_scorecard.png (dashboard moderno 2x3 com 16 lutadores e 87.5% de acurácia)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw

os.makedirs("docs/images", exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

CLUSTER_COLORS = {
    0: "#E67E22",  # Low Output / Weak Defense
    1: "#2980B9",  # Striker
    2: "#27AE60",  # Wrestler / Grappler
    3: "#8E44AD",  # Submission Specialist
}

CLUSTER_NAMES = {
    0: "Low Output / Weak Defense",
    1: "Striker",
    2: "Wrestler / Grappler",
    3: "Submission Specialist",
}

# -------------------------------------------------------------
# 1. CLUSTER PCA SCATTER PLOT COM 5 EXEMPLOS ARQUETÍPICOS
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

# Plota cada cluster
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

# Centroides
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

# 5 Lutadores Exemplares:
# Striker, Well-rounded (GSP), Wrestler, Sub. Specialist e Cluster 0 (CM Punk)
highlight_fighters = [
    ("Alex Pereira", "Alex Pereira (Striker)", (-110, 20), "#1B4F72"),
    ("Khabib Nurmagomedov", "Khabib Nurmagomedov (Wrestler / Grappler)", (-20, 65), "#145A32"),
    ("Georges St-Pierre", "Georges St-Pierre (Well-rounded)", (55, -40), "#D97706"),
    ("CM Punk", "CM Punk (Low Output / Weak Defense)", (-85, -45), "#D35400"),
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
            fontsize=8.5,
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

for ax_item, (metric, label) in zip(axes.flatten(), metrics_to_plot):
    ax_item.set_facecolor("#FAFAFA")
    ax_item.grid(axis="x", linestyle="--", alpha=0.5, color="#D5D8DC")
    
    bars = ax_item.barh(
        [CLUSTER_NAMES[c] for c in df_profile["cluster"]],
        df_profile[metric],
        color=[CLUSTER_COLORS[c] for c in df_profile["cluster"]],
        edgecolor="#333333",
        linewidth=0.8,
        height=0.6,
    )
    ax_item.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax_item.tick_params(axis="both", labelsize=9)
    
    for bar in bars:
        width = bar.get_width()
        ax_item.text(
            width + (max(df_profile[metric]) * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_item.set_xlim(0, max(df_profile[metric]) * 1.22)

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
# 4. DASHBOARD MODERNO DO GOLDEN SET (20 LUTADORES + PAINEL DE AUDITORIA)
# -------------------------------------------------------------
print("Gerando docs/images/golden_set_scorecard.png (Design UFC Octagon + Painel Geral de Auditoria)...")
with open("reports/golden_set_validation.json", "r", encoding="utf-8") as f:
    fighters_list = json.load(f)

def get_circular_avatar(fighter_name, size=128, border_color="#3B82F6", border_width=4):
    clean_name = fighter_name.replace(' "Poatan"', '').lower().replace(" ", "_")
    path = f"docs/images/fighters/{clean_name}.jpg"
    if not os.path.exists(path):
        fallback = Image.new("RGBA", (size, size), (30, 41, 59, 255))
        d = ImageDraw.Draw(fallback)
        d.ellipse((border_width, border_width, size - border_width, size - border_width), fill=(51, 65, 85))
        return fallback

    raw_img = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((border_width, border_width, size - border_width, size - border_width), fill=255)

    out_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_out = ImageDraw.Draw(out_img)
    draw_out.ellipse((0, 0, size, size), fill=border_color)

    raw_img.putalpha(mask)
    out_img.paste(raw_img, (0, 0), raw_img)
    return out_img

total_fighters = len(fighters_list)
total_correct = sum(1 for f in fighters_list if f["correct"])
acc_pct = (total_correct / total_fighters) * 100

fig = plt.figure(figsize=(16.0, 16.5), dpi=300)
fig.patch.set_facecolor("#080D1A")  # Deep UFC Octagon Dark
ax = fig.add_subplot(111)
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Banner Superior
header = FancyBboxPatch((2.0, 92.6), 96.0, 6.4, boxstyle="round,pad=0.5,rounding_size=1.0",
                        facecolor="#111A2E", edgecolor="#243350", linewidth=1.1)
ax.add_patch(header)

accent_bar = Rectangle((2.5, 98.5), 95.0, 0.5, facecolor="#DC2626", edgecolor="none")
ax.add_patch(accent_bar)

ax.text(4.2, 96.3, "UFC FIGHTING STYLE — GOLDEN SET AUDIT & LIMITAÇÕES DE DADOS",
        fontsize=14.5, fontweight="bold", color="#F8FAFC", va="center")
ax.text(4.2, 94.2, f"Auditoria supervisionada de {total_fighters} atletas históricos • Cartel oficial, predição de estilo e análise de dados ausentes",
        fontsize=8.6, color="#94A3B8", va="center")

# Badge de Acurácia no Topo
acc_badge = FancyBboxPatch((76.5, 93.3), 20.5, 5.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                           facecolor="#059669", edgecolor="#10B981", linewidth=1.1)
ax.add_patch(acc_badge)
ax.text(86.75, 96.2, f"{acc_pct:.1f}% ACURÁCIA", fontsize=11.0, fontweight="bold", color="#FFFFFF", ha="center", va="center")
ax.text(86.75, 94.3, f"{total_correct} de {total_fighters} Acertos (✓)", fontsize=8.2, color="#D1FAE5", ha="center", va="center")

# Layout dos 6 cards
layout_cards = [
    # Linha 1 (y = 71.4, h = 20.0) - 4 atletas cada
    {
        "title": "STRIKER",
        "color": "#38BDF8",
        "fighters": [f for f in fighters_list if f["expected"] == "Striker"],
        "x": 2.0, "y": 71.4, "w": 47.0, "h": 20.0,
        "step": 4.15, "top_pad": 4.3
    },
    {
        "title": "WRESTLER",
        "color": "#34D399",
        "fighters": [f for f in fighters_list if f["expected"] == "Wrestler"],
        "x": 51.0, "y": 71.4, "w": 47.0, "h": 20.0,
        "step": 4.15, "top_pad": 4.3
    },
    # Linha 2 (y = 58.8, h = 11.4) - 2 atletas cada
    {
        "title": "SUBMISSION SPECIALIST",
        "color": "#A78BFA",
        "fighters": [f for f in fighters_list if f["expected"] == "Submission Specialist"],
        "x": 2.0, "y": 58.8, "w": 47.0, "h": 11.4,
        "step": 4.25, "top_pad": 4.2
    },
    {
        "title": "LOW OUTPUT / WEAK DEFENSE",
        "color": "#FB923C",
        "fighters": [f for f in fighters_list if f["expected"] == "Low Output / Weak Defense"],
        "x": 51.0, "y": 58.8, "w": 47.0, "h": 11.4,
        "step": 4.25, "top_pad": 4.2
    },
    # Linha 3 (y = 31.0, h = 26.6) - 5 atletas / 3 atletas
    {
        "title": "WELL-ROUNDED (HÍBRIDOS)",
        "color": "#FBBF24",
        "fighters": [f for f in fighters_list if f["expected"] == "Well-rounded"],
        "x": 2.0, "y": 31.0, "w": 47.0, "h": 26.6,
        "step": 4.35, "top_pad": 4.3
    },
    {
        "title": "GRAPPLER / BJJ",
        "color": "#E2E8F0",
        "fighters": [f for f in fighters_list if f["expected"] == "Grappler"],
        "x": 51.0, "y": 31.0, "w": 47.0, "h": 26.6,
        "step": 4.35, "top_pad": 4.3
    },
]

for card in layout_cards:
    x0, y0, w, h = card["x"], card["y"], card["w"], card["h"]
    f_list = card["fighters"]
    c_correct = sum(1 for f in f_list if f["correct"])
    accent = card["color"]
    step = card["step"]
    top_pad = card["top_pad"]

    card_bg = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.3,rounding_size=0.9",
                             facecolor="#121A2B", edgecolor="#243350", linewidth=1.1)
    ax.add_patch(card_bg)

    edge_marker = Rectangle((x0, y0 + 1.0), 0.7, h - 2.0, facecolor=accent, edgecolor="none")
    ax.add_patch(edge_marker)

    card_title = f"{card['title']}  •  {c_correct}/{len(f_list)} ACERTOS"
    ax.text(x0 + 2.2, y0 + h - 1.8, card_title, fontsize=8.8, fontweight="bold", color=accent, va="center")

    divider = Rectangle((x0 + 1.8, y0 + h - 3.0), w - 3.6, 0.12, facecolor="#1E293B", edgecolor="none")
    ax.add_patch(divider)

    for i, f in enumerate(f_list):
        item_y = y0 + h - top_pad - (i * step)
        name = f["name"].replace(' "Poatan"', '')
        cartel = f.get("cartel", "N/A")
        pred = f["predicted"]

        if "Well-rounded" in pred:
            if "Wrestler / Striker" in pred:
                pred_short = "Well-rd. (Wrest/Strik)"
            elif "Striker / Wrestler" in pred:
                pred_short = "Well-rd. (Strik/Wrest)"
            else:
                pred_short = "Well-rounded"
        elif "Submission Specialist" in pred:
            pred_short = "Sub. Spec. (Grappler)"
        elif "Low Output" in pred:
            pred_short = "Low Output / Def."
        elif "Wrestler / Grappler" in pred:
            pred_short = "Wrestler / Grap."
        else:
            pred_short = pred

        avatar = get_circular_avatar(f["name"], size=128, border_color=accent, border_width=4)
        imagebox = OffsetImage(avatar, zoom=0.25)
        ab = AnnotationBbox(imagebox, (x0 + 3.6, item_y), frameon=False)
        ax.add_artist(ab)

        ax.text(x0 + 6.6, item_y + 0.85, name, fontsize=8.8, fontweight="bold", color="#F8FAFC", va="center")
        ax.text(x0 + 6.6, item_y - 0.95, f"Cartel: {cartel}", fontsize=7.4, color="#94A3B8", va="center")

        pill_w = 17.8
        pill_h = 2.6
        pill_x = x0 + 22.0
        pill_y = item_y - (pill_h / 2.0)
        pill_bg = FancyBboxPatch((pill_x, pill_y), pill_w, pill_h, boxstyle="round,pad=0.2,rounding_size=0.5",
                                 facecolor="#0B132B", edgecolor="#243350", linewidth=0.8)
        ax.add_patch(pill_bg)
        ax.text(pill_x + (pill_w / 2.0), item_y, f"→ {pred_short}", fontsize=7.9, fontweight="semibold",
                color="#93C5FD", ha="center", va="center")

        icon_w = 2.7
        icon_h = 2.6
        icon_x = x0 + 42.0
        icon_y = item_y - (icon_h / 2.0)
        if f["correct"]:
            badge = FancyBboxPatch((icon_x, icon_y), icon_w, icon_h, boxstyle="round,pad=0.2,rounding_size=0.6",
                                   facecolor="#064E3B", edgecolor="#10B981", linewidth=1.0)
            ax.add_patch(badge)
            ax.text(icon_x + (icon_w / 2.0), item_y, "✓", fontsize=11.5, fontweight="bold", color="#6EE7B7", ha="center", va="center")
        else:
            badge = FancyBboxPatch((icon_x, icon_y), icon_w, icon_h, boxstyle="round,pad=0.2,rounding_size=0.6",
                                   facecolor="#7F1D1D", edgecolor="#EF4444", linewidth=1.0)
            ax.add_patch(badge)
            ax.text(icon_x + (icon_w / 2.0), item_y, "✗", fontsize=11.5, fontweight="bold", color="#FCA5A5", ha="center", va="center")

    # Mini-callout no Card GRAPPLER
    if card["title"].startswith("GRAPPLER"):
        bjj_note = FancyBboxPatch((x0 + 2.2, y0 + 1.2), w - 4.4, 7.8, boxstyle="round,pad=0.2,rounding_size=0.6",
                                  facecolor="#0B132B", edgecolor="#334155", linewidth=0.8)
        ax.add_patch(bjj_note)
        ax.text(x0 + 3.6, y0 + 7.4, "DIAGNÓSTICO BJJ (1/3 ACERTOS):", fontsize=7.6, fontweight="bold", color="#E2E8F0", va="center")
        ax.text(x0 + 3.6, y0 + 5.5, "• Demian Maia e Gilbert Burns: BJJ mundial forçado a lutar em pé", fontsize=6.9, color="#94A3B8", va="center")
        ax.text(x0 + 3.6, y0 + 3.8, "• Falta de 'Control Time': médias caem no centroide Striker", fontsize=6.9, color="#FCA5A5", va="center")
        ax.text(x0 + 3.6, y0 + 2.1, "• Veja painel detalhado de dados faltantes abaixo ↓", fontsize=6.7, color="#64748B", va="center")

# =========================================================================
# PAINEL GERAL DE AUDITORIA & DADOS FALTANTES (FULL WIDTH AT BOTTOM)
# =========================================================================
panel_y = 2.4
panel_h = 27.4
main_panel = FancyBboxPatch((2.0, panel_y), 96.0, panel_h, boxstyle="round,pad=0.4,rounding_size=1.0",
                            facecolor="#0F172A", edgecolor="#243350", linewidth=1.2)
ax.add_patch(main_panel)

# Top accent bar do painel geral (Cyan accent)
panel_accent = Rectangle((2.5, panel_y + panel_h - 0.4), 95.0, 0.4, facecolor="#0284C7", edgecolor="none")
ax.add_patch(panel_accent)

# Título do Painel Geral
ax.text(4.2, panel_y + panel_h - 1.8, "PAINEL DE AUDITORIA TÉCNICA — PERFORMANCE POR ESTILO E GAPS DE DADOS DO DATASET",
        fontsize=10.2, fontweight="bold", color="#F8FAFC", va="center")

# 3 Colunas Internas no Painel Geral
cols = [
    {
        "title": "O QUE FUNCIONA MELHOR",
        "tag": "ALTA PRECISÃO (85-100%)",
        "color": "#10B981",
        "bg_tag": "#064E3B",
        "x": 3.6, "y": panel_y + 1.1, "w": 30.0, "h": panel_h - 3.4,
        "items": [
            ("Striker & Low Output", "Métricas por minuto (SLpM, SApM, Acc, Def) possuem altíssima granularidade e separam nocauteadores de alvos passivos com 100% de precisão nos perfis típicos do UFC."),
            ("Wrestlers Puros", "Lutadores com volume contínuo de quedas e pressão física (Khabib, Askren, Coleman) formam um grupo coeso e isolado no espaço projetado pelo PCA."),
            ("Well-rounded Híbridos", "A fronteira euclidiana (|d_strik - d_wrest| < 0.5) combinada à dupla competência resolve com precisão a transição suave de atletas modernos (GSP, Jones, Usman, Makhachev)."),
        ]
    },
    {
        "title": "ONDE O MODELO FALHA",
        "tag": "DIVERGÊNCIAS AUDITADAS",
        "color": "#EF4444",
        "bg_tag": "#7F1D1D",
        "x": 35.0, "y": panel_y + 1.1, "w": 30.0, "h": panel_h - 3.4,
        "items": [
            ("Grapplers de BJJ (33% acerto)", "Demian Maia e Gilbert Burns passaram rounds inteiros trocando em pé contra wrestlers defensivos (Woodley, Usman). Suas médias de solo diluíram para Striker."),
            ("Brawlers Ofensivos (Tuivasa)", "Pesos-pesados que absorvem muito dano (4.98 SApM) para contra-golpear têm diferencial de golpes negativo, caindo falsamente no Cluster 0 (Low Output)."),
            ("Currículo vs Ação Real (Romero)", "Medalhista olímpico de wrestling que no UFC quase não aplicava quedas (<1.8/15min) e só trocou em pé. O algoritmo avaliou a ação no octógono, não a medalha."),
        ]
    },
    {
        "title": "DADOS FALTANTES NO DATASET",
        "tag": "UFCSTATS GAPS (O QUE FALTA)",
        "color": "#F59E0B",
        "bg_tag": "#78350F",
        "x": 66.4, "y": panel_y + 1.1, "w": 30.0, "h": panel_h - 3.4,
        "items": [
            ("Tempo de Controle (Control Time)", "Duração em solo e grade. Resolveria imediatamente os casos de Demian Maia e Gilbert Burns, que controlam no chão sem necessariamente finalizar todo round."),
            ("Power Index & Knockdowns", "Potência e impacto do golpe. Diferenciaria brawlers nocauteadores resistentes (Tuivasa) de lutadores sem pegada e passivos (CM Punk, Moutinho)."),
            ("Distância vs Clinch vs Solo", "Separação dos golpes desferidos em pé, no dirty boxing/grade e no ground-and-pound. Permitiria clusterizar wrestlers de grade vs strikers à distância."),
        ]
    },
]

for col in cols:
    cx, cy, cw, ch = col["x"], col["y"], col["w"], col["h"]

    # Fundo da Sub-coluna
    sub_bg = FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.2,rounding_size=0.6",
                            facecolor="#121D33", edgecolor="#243350", linewidth=0.9)
    ax.add_patch(sub_bg)

    # Header da Sub-coluna
    ax.text(cx + 1.2, cy + ch - 1.4, col["title"], fontsize=8.4, fontweight="bold", color="#F8FAFC", va="center")

    # Tag de status
    tag_bg = FancyBboxPatch((cx + cw - 12.8, cy + ch - 2.2), 11.8, 1.7, boxstyle="round,pad=0.1,rounding_size=0.4",
                            facecolor=col["bg_tag"], edgecolor=col["color"], linewidth=0.7)
    ax.add_patch(tag_bg)
    ax.text(cx + cw - 6.9, cy + ch - 1.35, col["tag"], fontsize=6.2, fontweight="bold", color="#FFFFFF", ha="center", va="center")

    # Linha divisória interna
    col_div = Rectangle((cx + 1.0, cy + ch - 2.6), cw - 2.0, 0.08, facecolor="#1E293B", edgecolor="none")
    ax.add_patch(col_div)

    # Itens explicativos com espaçamento calculado para caber perfeitamente no card
    item_y_sub = cy + ch - 3.9
    for title, desc in col["items"]:
        ax.text(cx + 1.2, item_y_sub, f"▶ {title}", fontsize=7.4, fontweight="bold", color=col["color"], va="center")
        item_y_sub -= 1.15
        words = desc.split(" ")
        lines = []
        cur_line = ""
        for w_word in words:
            if len(cur_line + " " + w_word) < 52:
                cur_line = (cur_line + " " + w_word).strip()
            else:
                lines.append(cur_line)
                cur_line = w_word
        if cur_line:
            lines.append(cur_line)

        for line_t in lines:
            ax.text(cx + 2.2, item_y_sub, line_t, fontsize=6.6, color="#94A3B8", va="center")
            item_y_sub -= 0.95
        item_y_sub -= 0.45

# Rodapé informativo
ax.text(50.0, 1.1, "Pipeline UFC Fighting Style • Apache Airflow DAG • K-Means (k=4) + PCA Dinâmico + Fronteira Geométrica",
        fontsize=8.0, color="#64748B", ha="center", va="center")

plt.tight_layout()
plt.savefig("docs/images/golden_set_scorecard.png", facecolor="#080D1A")
plt.close()

print("Todas as 4 imagens foram geradas com sucesso em docs/images/!")


