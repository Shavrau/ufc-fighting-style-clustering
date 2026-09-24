# Como rodar o pipeline no Apache Airflow

Pré-requisitos: Docker Engine rodando no WSL (sem necessidade de Docker Desktop),
`docker compose` disponível.

## 1. Coloque o dataset no lugar certo

```bash
mkdir -p data/raw
# copie ufc-fighters-statistics.csv para data/raw/
```

## 2. Configure o `.env`

Copie o template e preencha com seus próprios valores:

```bash
cp .env.example .env
```

Gere uma `SECRET_KEY` fixa (importante — sem isso, o Airflow gera uma nova aleatória a
cada restart do container, invalidando sessões de login e causando erro de CSRF):

```bash
python3 -c "import secrets; print(secrets.token_hex(16))"
```

Cole o resultado em `AIRFLOW__WEBSERVER__SECRET_KEY` no `.env`.

## 3. Confirme os volumes no `docker-compose.yaml`

Dentro do bloco `x-airflow-common` → `volumes`, garanta que existem estas linhas (além
das padrão de `dags`, `logs`, `plugins`, `config`):

```yaml
- ${AIRFLOW_PROJ_DIR:-.}/src:/opt/airflow/src
- ${AIRFLOW_PROJ_DIR:-.}/data:/opt/airflow/data
- ${AIRFLOW_PROJ_DIR:-.}/models:/opt/airflow/models
- ${AIRFLOW_PROJ_DIR:-.}/reports:/opt/airflow/reports
```

## 4. Build e subida

```bash
docker compose build
docker compose up airflow-init   # só na primeira vez
docker compose up -d
```

Espere 1-2 minutos para o webserver ficar saudável (`docker compose ps` para conferir).

## 5. Acesse a UI

`http://localhost:8080` — login padrão `airflow` / `airflow` (a menos que tenha
customizado no `.env`).

## 6. Ative e dispare a DAG

Procure `fighting_style_clustering` na lista, ative o toggle (DAGs começam pausadas por
padrão), clique em ▶ → "Trigger DAG".

## 7. Acompanhe e confira os resultados

Na Grid View, clique em cada task para ver os logs. Ao final, confira (no seu host, não
só dentro do container, já que os volumes são compartilhados):

- `data/processed/fighters_clustered_labeled.parquet`
- `models/kmeans.pkl`, `models/scaler.pkl`, `models/pca.pkl`
- `reports/cluster_profile.csv`, `reports/cluster_labels.json`

## Rodando os scripts fora do Airflow (debug local)

Todo script em `src/` roda isolado, sem precisar do Airflow no ar:

```bash
python3 src/preprocessing.py
python3 src/feature_engineering.py
python3 src/train_model.py
python3 src/evaluate_model.py
python3 src/scoring.py   # busca por nome/apelido no prompt ou via: python3 src/scoring.py "Alex Pereira"
python3 src/validate_golden_set.py   # roda a validação com 12 lutadores conhecidos (acurácia 83%)
```

## Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `Bad Request: CSRF tokens do not match` | `SECRET_KEY` não fixada, mudou entre restarts | Ver passo 2; limpar cookies/aba anônima como paliativo |
| `ResolutionImpossible` no `pip install` do build | Versão fixada no `requirements.txt` conflita com o constraints file do Airflow | Ver a versão exigida na mensagem de erro e alinhar, ou soltar o pino (`>=`) |
