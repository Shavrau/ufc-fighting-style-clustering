FROM apache/airflow:2.7.2-python3.11

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.11.txt"
