"""Application entry-point: creates the FastAPI app and wires everything up."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from segmentation_service import __version__
from segmentation_service.api.router import root_router
from segmentation_service.config import AppEnv, get_settings
from segmentation_service.logging_config import LogContext, configure_logging, get_logger

_WEB_DIR = Path(__file__).resolve().parent / "web"

settings = get_settings()
configure_logging(settings.log_level.value)
log = LogContext(get_logger(__name__))


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info(
        "Service starting",
        env=settings.app_env.value,
        backend=settings.segmentation_backend.value,
        version=__version__,
    )
    yield
    log.info("Service shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stable Segmentation Service",
        description="Modular segmentation inference API with pluggable backends.",
        version=__version__,
        lifespan=_lifespan,
        docs_url="/docs" if settings.app_env != AppEnv.production else None,
        redoc_url="/redoc" if settings.app_env != AppEnv.production else None,
    )

    # ---- middleware ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- routers ----
    app.include_router(root_router, prefix=settings.api_prefix)

    # ---- demo UI — served at /demo (html=True serves index.html for /) ----
    app.mount("/demo", StaticFiles(directory=str(_WEB_DIR), html=True), name="demo")

    return app


app = create_app()


def start() -> None:
    """Entrypoint used by the `serve` script in pyproject.toml."""
    uvicorn.run(
        "segmentation_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == AppEnv.development,
    )


if __name__ == "__main__":
    start()
