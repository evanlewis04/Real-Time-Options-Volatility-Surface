"""Local API-shaped service for the filings intelligence workbench."""

from .local_service import (
    LocalRagApiService,
    LocalApiError,
    QueryRequest,
    api_endpoint_manifest,
    build_local_api_service,
    call_local_api_endpoint,
    create_fastapi_app,
    serialize_chunk,
    serialize_coverage_report,
    serialize_query_result,
)
from .smoke import DEFAULT_PHASE7_QUERY, ApiSmokeReport, ApiSmokeStep, run_api_smoke

__all__ = [
    "DEFAULT_PHASE7_QUERY",
    "ApiSmokeReport",
    "ApiSmokeStep",
    "LocalRagApiService",
    "LocalApiError",
    "QueryRequest",
    "api_endpoint_manifest",
    "build_local_api_service",
    "call_local_api_endpoint",
    "create_fastapi_app",
    "serialize_chunk",
    "serialize_coverage_report",
    "serialize_query_result",
    "run_api_smoke",
]
