import os
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from dotenv import load_dotenv

from cdss.heliot.config_loader import load_config

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parents[3]
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI
from .heliot_endpoints import router as api_router
from fastapi.middleware.cors import CORSMiddleware

def get_redis_url() -> str:
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is not set")
    return redis_url


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load application config once at startup
    app.state.config = load_config()

    # Create shared Redis client once at startup
    redis_client = redis.from_url(get_redis_url(), decode_responses=False)
    app.state.redis = redis_client

    try:
        yield
    finally:
        await redis_client.aclose()

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
