SET /p job_id="Digite o nome do Job: "
CALL gcloud ai-platform jobs submit training %job_id% --module-name=trainer.treinar_rede_particulas_cristais --package-path=./ParticulasCristaisDeGelo/trainer --job-dir=gs://cloud-computing-keras-am/ParticulasCristaisDeGelo --region=us-central1 --config=ParticulasCristaisDeGelo/trainer/cloudml-gpu.yaml
CALL gcloud ai-platform jobs stream-logs %job_id%
PAUSE