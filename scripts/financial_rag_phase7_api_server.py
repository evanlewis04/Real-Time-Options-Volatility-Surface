"""Launch the optional local FastAPI adapter for Phase 7."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.api import api_endpoint_manifest, build_local_api_service, create_fastapi_app
from src.financial_rag.settings import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local Financial RAG API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port. Default: 8765.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install optional API server dependencies to launch HTTP.")
        return 2

    service = build_local_api_service(root=project_root(), use_voyage=False)
    health = service.health()
    print(f"Financial RAG local API target: http://{args.host}:{args.port}")
    print(f"Cache: chunks={health['chunk_count']} embeddings={health['embedding_count']} status={health['status']}")
    print("Endpoints:")
    for endpoint in api_endpoint_manifest():
        print(f"- {endpoint['method']} {endpoint['path']} - {endpoint['description']}")
    print("Provider behavior: cached local chunks/vectors only; no SEC refetch or LLM calls.")
    try:
        app = create_fastapi_app(service)
    except RuntimeError as exc:
        print(str(exc))
        print("The local service and smoke scripts still work without FastAPI.")
        print("Install FastAPI to expose HTTP endpoints.")
        return 2

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
