"""FastAPI OpenAI-compatible API server for Zensei models.

Provides:
  - POST /v1/chat/completions  (streaming and non-streaming)
  - GET  /v1/models            (list available models)
  - GET  /health               (health check)

Usage:
    python -m zensei.serving.api_server --model_path checkpoints/zensei-16b --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Union

import fire
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zensei.model.model import ModelArgs, Transformer
from zensei.serving.chat_template import apply_chat_template
from zensei.serving.generate import generate, generate_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models (OpenAI API compatible)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "zensei"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stop: Optional[list[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "zensei"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model: Optional[Transformer] = None
_tokenizer = None
_model_name: str = "zensei"
_semaphore: Optional[asyncio.Semaphore] = None


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


def _load_model(model_path: str, tokenizer_path: Optional[str] = None, device: str = "cuda"):
    """Load model and tokenizer into global state."""
    global _model, _tokenizer, _model_name

    from transformers import AutoTokenizer
    import os

    tokenizer_path = tokenizer_path or model_path
    logger.info("Loading tokenizer from %s", tokenizer_path)
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    logger.info("Loading model from %s", model_path)
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        _model = Transformer.from_config(config_path)
    else:
        logger.warning("No config.json found in %s; using default ModelArgs.", model_path)
        _model = Transformer(ModelArgs())

    # Load weights if available
    weight_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        _model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights from %s", weight_path)
    else:
        logger.warning("No weights found; model is randomly initialized.")

    _model = _model.to(device).eval()
    _model_name = os.path.basename(model_path.rstrip("/"))
    logger.info("Model loaded: %s (device=%s)", _model_name, device)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Zensei API Server",
    description="OpenAI-compatible API for Zensei Japanese language models",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": _model_name,
        "model_loaded": _model is not None,
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelListResponse(
        data=[
            ModelInfo(
                id=_model_name,
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (SSE) and non-streaming responses.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if _semaphore is not None:
        acquired = _semaphore.locked()
        if acquired and _semaphore._value == 0:
            raise HTTPException(status_code=429, detail="Too many concurrent requests")

    # Convert messages to chat template format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt = apply_chat_template(messages, add_generation_prompt=True)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_response(
                prompt=prompt,
                request_id=request_id,
                created=created,
                model=request.model,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            ),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_response(
            prompt=prompt,
            request_id=request_id,
            created=created,
            model=request.model,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )


async def _stream_response(
    prompt: str,
    request_id: str,
    created: int,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> AsyncGenerator[str, None]:
    """Generate streaming SSE response."""
    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Generate tokens
    for token_str in generate_stream(
        _model,
        _tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    ):
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=token_str),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        # Allow other coroutines to run
        await asyncio.sleep(0)

    # Send final chunk
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _non_stream_response(
    prompt: str,
    request_id: str,
    created: int,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> ChatCompletionResponse:
    """Generate a non-streaming response."""
    prompt_tokens = len(_tokenizer.encode(prompt))

    output_text = generate(
        _model,
        _tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    completion_tokens = len(_tokenizer.encode(output_text))

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cuda",
    max_concurrent: int = 4,
) -> None:
    """Start the Zensei API server.

    Args:
        model_path: Path to the model checkpoint directory.
        tokenizer_path: Path to the tokenizer (defaults to model_path).
        host: Host to bind the server to.
        port: Port to listen on.
        device: Device to use (cuda or cpu).
        max_concurrent: Maximum number of concurrent generation requests.
    """
    global _semaphore

    _load_model(model_path, tokenizer_path, device)
    _semaphore = asyncio.Semaphore(max_concurrent)

    logger.info("Starting Zensei API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    fire.Fire(main)
