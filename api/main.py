"""
# NOTE: Claude offline tokenizer sağlamıyor. Bu yüzden eklenmedi.

# TODO: Anasayfaya model ekle butonu ekle.
# TODO: Uygulamayı internette canlıya al.
# TODO: Linting yap.
"""


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from db import init_db, add_model, delete_model, list_models
from contextlib import asynccontextmanager
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict
import shutil
import tiktoken
import os


tiktoken_cache_dir = "models/tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

MODELS_CACHE_DIR = Path("models")
GPT_MODELS = ["gpt-4o", "gpt-4.1", "gpt-5"]
tokenizer_cache: Dict[str, object] = {}


def get_or_load_tokenizer(model_name: str):
    tokenizer = tokenizer_cache.get(model_name)
    if tokenizer is None:
        if model_name.lower() in GPT_MODELS:
            tokenizer = tiktoken.encoding_for_model(model_name.lower())

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODELS_CACHE_DIR)+"/"+model_name, token="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")

        tokenizer_cache[model_name] = tokenizer

    return tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    MODELS_CACHE_DIR.mkdir(exist_ok=True)

    for row in list_models():
        name = row.get("name")
        if not name:
            continue

        try:
            get_or_load_tokenizer(name)

        except Exception:
            pass

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CountRequest(BaseModel):
    model: str
    text: str


class CountResponse(BaseModel):
    model: str
    tokens: int


class AddModelRequest(BaseModel):
    name: str


class DeleteModelRequest(BaseModel):
    name: str


@app.post("/api/count", response_model=CountResponse)
async def count_tokens_api(payload: CountRequest) -> CountResponse:
    try:
        tokenizer = tokenizer_cache[payload.model]
        tokens = len(tokenizer.encode(payload.text))

        return CountResponse(model=payload.model, tokens=tokens)

    except Exception:
        raise HTTPException(status_code=404, detail="Geçersiz model.")


@app.post("/models/add")
async def add_model_form(payload: AddModelRequest):
    if payload.name in tokenizer_cache:
        raise HTTPException(status_code=400, detail="Model zaten mevcut.")

    try:
        get_or_load_tokenizer(payload.name)
        add_model(name=payload.name)
    
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz model.")
    
    return {"ok": True}


@app.post("/models/delete")
async def delete_model_form(payload: DeleteModelRequest):
    delete_model(name=payload.name)
    tokenizer_cache.pop(payload.name, None)

    try:
        model_dir = MODELS_CACHE_DIR / payload.name
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    except Exception:
        pass

    return {"ok": True}


@app.get("/models/list")
async def list_models_form():
    return list_models()

# Serve static UI
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=27275)