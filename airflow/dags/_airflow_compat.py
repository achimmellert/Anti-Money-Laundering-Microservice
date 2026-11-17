"""
Airflow Decorator Compatibility Layer
=====================================
Uses native `apache-airflow` TaskFlow decorators (`@dag`, `@task`, `@task.branch`)
when running inside the Airflow scheduler/webserver container, while providing a
lightweight local execution shim so all ML Engineer DAG tasks can be imported and
unit-tested directly in `pytest` without requiring the full Airflow runtime in the
microservice virtual environment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    from airflow.decorators import dag as airflow_dag  # type: ignore[import-not-found]
    from airflow.decorators import task as airflow_task  # type: ignore[import-not-found]

    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False

    def airflow_dag(*dag_args: Any, **dag_kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func._is_airflow_dag = True  # type: ignore[attr-defined]
            func._dag_kwargs = dag_kwargs  # type: ignore[attr-defined]
            return func

        return _decorator

    class _TaskDecoratorShim:
        def __call__(self, *task_args: Any, **task_kwargs: Any) -> Any:
            if len(task_args) == 1 and callable(task_args[0]) and not task_kwargs:
                fn = task_args[0]
                fn.function = fn  # type: ignore[attr-defined]
                return fn

            def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                func.function = func  # type: ignore[attr-defined]
                return func

            return _decorator

        def branch(self, *task_args: Any, **task_kwargs: Any) -> Any:
            return self.__call__(*task_args, **task_kwargs)

    airflow_task = _TaskDecoratorShim()

__all__ = ["AIRFLOW_INSTALLED", "airflow_dag", "airflow_task"]
