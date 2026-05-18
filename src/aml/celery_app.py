from celery import Celery
from celery.signals import worker_process_init
from aml.config.settings import settings


celery_app = Celery(
    "aml_tasks",
    broker=settings.REDIS_URL.unicode_string(),
    backend=settings.REDIS_URL.unicode_string(),
    include=[
        "aml.tasks.enrichment",
        "aml.tasks.inference",
        "aml.tasks.persistence"
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        'aml.tasks.enrichment.run_enrichment_task': {'queue': 'enrichment'},
        'aml.tasks.inference.run_inference_task': {'queue': 'inference'},
        'aml.tasks.inference.generate_shap_task': {'queue': 'background'},
        'aml.tasks.persistence.run_persistence_task': {'queue': 'persistence'},
    }
)


@worker_process_init.connect
def init_worker_db_pool(**kwargs):
    from aml.services.database import engine
    engine.dispose()
