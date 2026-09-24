"""
scoring.py

Scoring step for the fighting-style clustering pipeline. Takes a single
fighter's raw stats and tells you which trained cluster (style group) they
belong to. Runs the exact same transform chain as training
(preprocessing.py -> feature_engineering.py) but only *applies* the already
-fitted scaler/PCA/K-Means instead of refitting them.

Usage from the command line (interactive prompts):
    python3 scoring.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DAGS_DIR = Path(__file__).resolve().parent.parent

SCALER_PATH = DAGS_DIR / "models" / "scaler.pkl"
PCA_PATH = DAGS_DIR / "models" / "pca.pkl"
KMEANS_PATH = DAGS_DIR / "models" / "kmeans.pkl"
LABELS_PATH = DAGS_DIR / "reports" / "cluster_labels.json"
LABELED_DATA_PATH = DAGS_DIR / "data" / "processed" / "fighters_clustered_labeled.parquet"
RAW_DATA_PATH = DAGS_DIR / "data" / "raw" / "ufc-fighters-statistics.csv"

STYLE_FEATURES = [
    "significant_strikes_landed_per_minute",
    "significant_striking_accuracy",
    "significant_strikes_absorbed_per_minute",
    "significant_strike_defence",
    "average_takedowns_landed_per_15_minutes",
    "takedown_accuracy",
    "takedown_defense",
    "average_submissions_attempted_per_15_minutes",
]

SKEWED_FEATURES = []

TD_WEIGHT = 1.5
SUB_WEIGHT = 2.5


def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    with open(LABELS_PATH) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return scaler, pca, kmeans, labels


def transform(stats: dict, scaler, pca) -> np.ndarray:
    row = pd.DataFrame([{f: stats[f] for f in STYLE_FEATURES}])

    for col in SKEWED_FEATURES:
        row[col] = np.log1p(row[col])

    scaled = scaler.transform(row[STYLE_FEATURES])
    scaled_cols = [f"{c}_scaled" for c in STYLE_FEATURES]
    scaled_df = pd.DataFrame(scaled, columns=scaled_cols)

    scaled_df["striking_net_scaled"] = (
        scaled_df["significant_strikes_landed_per_minute_scaled"]
        - scaled_df["significant_strikes_absorbed_per_minute_scaled"]
    )
    scaled_df["grappling_pressure_scaled"] = (
        scaled_df["average_takedowns_landed_per_15_minutes_scaled"]
        + scaled_df["average_submissions_attempted_per_15_minutes_scaled"]
    ) / 2
    scaled_df["defensive_rating_scaled"] = (
        scaled_df["significant_strike_defence_scaled"] + scaled_df["takedown_defense_scaled"]
    ) / 2

    feature_cols = scaled_cols + ["striking_net_scaled", "grappling_pressure_scaled", "defensive_rating_scaled"]
    scaled_df["average_takedowns_landed_per_15_minutes_scaled"] *= TD_WEIGHT
    scaled_df["takedown_accuracy_scaled"] *= TD_WEIGHT
    scaled_df["average_submissions_attempted_per_15_minutes_scaled"] *= SUB_WEIGHT
    scaled_df["grappling_pressure_scaled"] *= (TD_WEIGHT + SUB_WEIGHT) / 2

    return pca.transform(scaled_df[feature_cols])


def score_fighter(stats: dict, artifacts=None) -> dict:
    scaler, pca, kmeans, labels = artifacts or load_artifacts()

    X = transform(stats, scaler, pca)
    cluster_id = int(kmeans.predict(X)[0])
    distances = kmeans.transform(X)[0]

    ranked = sorted(
        [{"cluster": i, "label": labels[i]["label"], "distance": round(float(d), 3)}
         for i, d in enumerate(distances)],
        key=lambda r: r["distance"],
    )

    borderline = (ranked[1]["distance"] - ranked[0]["distance"]) < 0.5
    base_label = labels[cluster_id]["label"]

    # Dynamic labeling: Check for Well-rounded / Hybrid profile
    sl = stats.get("significant_strikes_landed_per_minute", 0)
    td = stats.get("average_takedowns_landed_per_15_minutes", 0)
    tdd = stats.get("takedown_defense", 0)

    c0 = ranked[0]["cluster"]
    c1 = ranked[1]["cluster"]

    # 1. Borderline between Striker (cluster 1) and Wrestler/Grappler (cluster 2)
    is_borderline_hybrid = borderline and (c0 in (1, 2) and c1 in (1, 2))
    # 2. Dual competency: active striking AND active wrestling with high takedown defense
    has_dual_skills = (sl >= 3.5 and 1.2 <= td < 4.5 and tdd >= 75.0)

    if (is_borderline_hybrid or has_dual_skills) and td < 4.5:
        p0 = ranked[0]["label"].split(" / ")[0]
        p1 = ranked[1]["label"].split(" / ")[0]
        final_label = f"Well-rounded ({p0} / {p1})"
    else:
        final_label = base_label

    return {
        "cluster": cluster_id,
        "base_cluster_label": base_label,
        "label": final_label,
        "distance_to_assigned_centroid": round(float(distances[cluster_id]), 3),
        "ranked_clusters": ranked,
        "borderline": borderline,
    }


def lookup_fighter(query: str, artifacts=None) -> list:
    """Finds fighters in the dataset by name or nickname and scores them."""
    query = query.strip()
    if not query:
        return []

    if LABELED_DATA_PATH.exists():
        df = pd.read_parquet(LABELED_DATA_PATH)
    elif RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        return []

    # Exact match on name
    exact = df[df["name"].str.lower() == query.lower()]
    if len(exact) == 1:
        matches = exact
    else:
        # Partial match on name or nickname
        mask = df["name"].str.contains(query, case=False, na=False)
        if "nickname" in df.columns:
            mask |= df["nickname"].str.contains(query, case=False, na=False)
        matches = df[mask]

    results = []
    for _, row in matches.iterrows():
        stats = {f: float(row[f]) for f in STYLE_FEATURES if f in row and pd.notna(row[f])}
        score = None
        if len(stats) == len(STYLE_FEATURES):
            # Check if cluster already exists in dataset
            if "cluster" in row and pd.notna(row["cluster"]) and "cluster_label" in row and pd.notna(row["cluster_label"]):
                score = score_fighter(stats, artifacts=artifacts)
            else:
                score = score_fighter(stats, artifacts=artifacts)

        results.append({
            "name": row.get("name", "Desconhecido"),
            "nickname": row.get("nickname", "") if pd.notna(row.get("nickname")) else "",
            "wins": int(row.get("wins", 0)) if pd.notna(row.get("wins")) else 0,
            "losses": int(row.get("losses", 0)) if pd.notna(row.get("losses")) else 0,
            "draws": int(row.get("draws", 0)) if pd.notna(row.get("draws")) else 0,
            "stats": stats,
            "score": score,
        })
    return results


def format_fighter_card(fighter: dict) -> str:
    name = fighter["name"]
    nick = f' "{fighter["nickname"]}"' if fighter["nickname"] else ""
    record = f'{fighter["wins"]} Vitórias - {fighter["losses"]} Derrotas - {fighter["draws"]} Empates'
    score = fighter.get("score")

    lines = [
        "=" * 64,
        f"🥊 {name}{nick}",
        f"📊 Cartel: {record}",
        "=" * 64,
    ]
    if score:
        cluster = score["cluster"]
        label = score["label"].upper()
        base_label = score.get("base_cluster_label", "")
        borderline_str = " (Atenção: Borderline / Características Híbridas)" if score["borderline"] else ""
        if base_label and base_label.upper() != label:
            lines.append(f"👉 Estilo de Luta: {label}")
            lines.append(f"   (Cluster base: {cluster} - {base_label}){borderline_str}")
        else:
            lines.append(f"👉 Estilo de Luta: {label} (Cluster {cluster}){borderline_str}")
        lines.append("-" * 64)

        st = fighter.get("stats", {})
        if st:
            lines.append("Métricas de Performance:")
            lines.append(f"  • Golpes significativos dados/min: {st.get('significant_strikes_landed_per_minute', 0):.2f} (Precisão: {st.get('significant_striking_accuracy', 0):.1f}%)")
            lines.append(f"  • Golpes absorvidos/min:           {st.get('significant_strikes_absorbed_per_minute', 0):.2f} (Defesa:   {st.get('significant_strike_defence', 0):.1f}%)")
            lines.append(f"  • Quedas a cada 15 min:            {st.get('average_takedowns_landed_per_15_minutes', 0):.2f} (Precisão: {st.get('takedown_accuracy', 0):.1f}%, Defesa: {st.get('takedown_defense', 0):.1f}%)")
            lines.append(f"  • Submissões tentadas a cada 15m:  {st.get('average_submissions_attempted_per_15_minutes', 0):.2f}")
            lines.append("-" * 64)

        lines.append(f"Distância ao centróide do estilo: {score['distance_to_assigned_centroid']}")
        rank_str = " | ".join(f"{r['label']}: {r['distance']}" for r in score["ranked_clusters"])
        lines.append(f"Afinidade com outros estilos: {rank_str}")
    else:
        lines.append("⚠️ Estatísticas de performance ausentes ou zeradas na base do UFC.")
    lines.append("=" * 64)
    return "\n".join(lines)


def prompt_for_stats() -> dict:
    print("Insira as estatísticas numéricas do lutador (unidades originais):\n")
    stats = {}
    for feature in STYLE_FEATURES:
        while True:
            raw = input(f"  {feature}: ").strip()
            try:
                stats[feature] = float(raw)
                break
            except ValueError:
                print("    por favor digite um número válido")
    return stats


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    artifacts = load_artifacts()

    # Se um nome foi passado diretamente pela linha de comando
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        matches = lookup_fighter(query, artifacts=artifacts)
        if not matches:
            print(f"\n❌ Nenhum lutador encontrado com o nome '{query}'.")
            sys.exit(1)
        elif len(matches) == 1:
            print("\n" + format_fighter_card(matches[0]))
        else:
            print(f"\nEncontrados {len(matches)} lutadores com '{query}':")
            for idx, m in enumerate(matches[:10], 1):
                nick = f' "{m["nickname"]}"' if m["nickname"] else ""
                lbl = m["score"]["label"] if m["score"] else "N/A"
                print(f"  [{idx}] {m['name']}{nick} -> {lbl}")
            if len(matches) > 10:
                print(f"  ... e mais {len(matches) - 10} resultados.")
        sys.exit(0)

    # Modo interativo simples
    print("=" * 64)
    print("🥊 UFC Fighting Style Identifier — Busca de Estilo de Luta")
    print("=" * 64)
    print("Digite o nome ou apelido de um lutador do UFC (ex: 'Alex Pereira', 'Poatan', 'Khabib')")
    print("Comandos: 'manual' para digitar estatísticas | 'sair' para encerrar\n")

    while True:
        try:
            query = input("Nome do lutador: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

        if not query:
            continue
        if query.lower() in ("sair", "exit", "quit", "q"):
            print("Até logo!")
            break
        if query.lower() == "manual":
            fighter_stats = prompt_for_stats()
            result = score_fighter(fighter_stats, artifacts=artifacts)
            print(f"\nEstilo previsto: cluster {result['cluster']} - {result['label']}")
            print(f"Distância ao centróide: {result['distance_to_assigned_centroid']}")
            if result["borderline"]:
                print("Nota: lutador borderline (próximo de mais de um estilo).")
            print("Distâncias a todos os clusters:")
            for r in result["ranked_clusters"]:
                print(f"  cluster {r['cluster']} ({r['label']}): {r['distance']}")
            print()
            continue

        matches = lookup_fighter(query, artifacts=artifacts)
        if not matches:
            print(f"❌ Nenhum lutador encontrado para '{query}'. Tente outro nome ou digite 'manual'.\n")
            continue

        if len(matches) == 1:
            print("\n" + format_fighter_card(matches[0]) + "\n")
        else:
            print(f"\nEncontrados {len(matches)} lutadores correspondentes:")
            for idx, m in enumerate(matches[:10], 1):
                nick = f' "{m["nickname"]}"' if m["nickname"] else ""
                rec = f"{m['wins']}V-{m['losses']}D"
                lbl = m["score"]["label"] if m["score"] else "Sem stats"
                print(f"  [{idx}] {m['name']}{nick} ({rec}) -> {lbl}")
            if len(matches) > 10:
                print(f"  ... ({len(matches) - 10} outros ocultados)")

            choice = input("\nEscolha um número para ver os detalhes (ou Enter para nova busca): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= min(len(matches), 10):
                print("\n" + format_fighter_card(matches[int(choice) - 1]) + "\n")
            else:
                print()
