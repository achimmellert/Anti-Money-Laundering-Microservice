# src/celery_app.py

from celery import Celery
from celery.signals import worker_process_init
from src.config.settings import settings

#import sentry_sdk
#from sentry_sdk.integrations.celery import CeleryIntegration


celery_app = Celery(
    "aml_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "src.tasks.enrichment",
        "src.tasks.inference",
        "src.tasks.persistence"
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
        'src.tasks.enrichment.*': {'queue': 'enrichment'},
        'src.tasks.inference.run_inference_task': {'queue': 'inference'},
        'src.tasks.inference.generate_shap_task': {'queue': 'inference'},  # Explicitly route the SHAP task
        'src.tasks.persistence.*': {'queue': 'persistence'},
    }
)

#sentry_sdk.init(
#    dsn=settings.SENTRY_DSN,
#    integrations=[CeleryIntegration()]
#)

@worker_process_init.connect
def init_worker_db_pool(**kwargs):
    from src.services.database import engine
    engine.dispose()