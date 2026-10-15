"""FastAPI wrapper around the investing agent.

Run (Windows or Linux):   uv run python agent_api.py
Linux/Docker also works:  uvicorn agent_api:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health
POST /assessments                      start a run; returns when the report awaits approval
POST /assessments/stream               same, as Server-Sent Events (progress + final result)
GET  /assessments/{thread_id}          current status / report (for page refresh, polling)
POST /assessments/{thread_id}/decision approve | reject (resumes the interrupted graph)
POST /assessments/{thread_id}/qna      ask questions about the saved report
"""
import asyncio
import json
import logging
import os
import selectors
import sys
from contextlib import asynccontextmanager
from typing import Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

import agent_chat as chat
from analysis.risk_engine import HORIZONS, PROFILES
from models.models import UserPrefs

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat.setup_logging()
    rt = chat.AgentRuntime()
    await rt.start()                 # MCP tools, LLMs, Postgres checkpointer, compiled graph: ONCE
    app.state.rt = rt
    app.state.in_flight = set()      # thread_ids currently running (blocks double-submits)
    log.info("API ready")
    try:
        yield
    finally:
        await rt.stop()


app = FastAPI(title="Investing Agent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class AssessmentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    profile: str
    horizon: str


class DecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    comment: Optional[str] = Field(default=None, max_length=1000)


class ChatTurn(BaseModel):
    role: Literal["user", "ai"]
    content: str = Field(max_length=4000)


class QnaRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rt(request: Request) -> chat.AgentRuntime:
    return request.app.state.rt


def _prefs(req: AssessmentRequest) -> UserPrefs:
    profile, horizon = req.profile.strip().lower(), req.horizon.strip().lower()
    if profile not in PROFILES:
        raise HTTPException(422, f"profile must be one of {list(PROFILES)}")
    if horizon not in HORIZONS:
        raise HTTPException(422, f"horizon must be one of {list(HORIZONS)}")
    try:
        return UserPrefs(profile=profile, horizon=horizon)
    except ValidationError as e:
        raise HTTPException(422, str(e))


def _claim(request: Request, thread_id: str) -> None:
    in_flight: set = request.app.state.in_flight
    if thread_id in in_flight:
        raise HTTPException(409, "This assessment is already being processed.")
    in_flight.add(thread_id)


def _release(request: Request, thread_id: str) -> None:
    request.app.state.in_flight.discard(thread_id)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "tools": len(_rt(request).tools)}


@app.post("/assessments")
async def create_assessment(req: AssessmentRequest, request: Request):
    rt, prefs = _rt(request), _prefs(req)
    thread_id = str(uuid4())
    _claim(request, thread_id)
    try:
        async for _ in chat.start_assessment_stream(rt, req.query.strip(), prefs, thread_id):
            pass
        return await chat.get_status(rt, thread_id)
    except Exception:
        log.exception("assessment %s failed", thread_id)
        raise HTTPException(500, f"Assessment failed (thread {thread_id}). See server logs.")
    finally:
        _release(request, thread_id)


@app.post("/assessments/stream")
async def create_assessment_stream(req: AssessmentRequest, request: Request):
    """SSE: `started` -> many `progress` -> `result` (or `error`). Use fetch() + ReadableStream on the frontend."""
    rt, prefs = _rt(request), _prefs(req)
    thread_id = str(uuid4())
    _claim(request, thread_id)

    async def gen():
        try:
            yield _sse("started", {"thread_id": thread_id})
            async for ev in chat.start_assessment_stream(rt, req.query.strip(), prefs, thread_id):
                yield _sse("progress", ev)
            yield _sse("result", await chat.get_status(rt, thread_id))
        except Exception:
            log.exception("assessment %s failed", thread_id)
            yield _sse("error", {"thread_id": thread_id, "message": "Assessment failed. See server logs."})
        finally:
            _release(request, thread_id)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/assessments/{thread_id}")
async def get_assessment(thread_id: str, request: Request):
    status = await chat.get_status(_rt(request), thread_id)
    if status["status"] == "not_found":
        raise HTTPException(404, "Unknown thread_id.")
    return status


@app.post("/assessments/{thread_id}/decision")
async def decide(thread_id: str, body: DecisionRequest, request: Request):
    rt = _rt(request)
    _claim(request, thread_id)
    try:
        status = await chat.get_status(rt, thread_id)
        if status["status"] == "not_found":
            raise HTTPException(404, "Unknown thread_id.")
        if status["status"] != "awaiting_approval":
            raise HTTPException(409, f"Nothing to approve: status is '{status['status']}'.")
        async for _ in chat.resume_assessment_stream(rt, thread_id, body.decision, body.comment):
            pass
        return await chat.get_status(rt, thread_id)
    except HTTPException:
        raise
    except Exception:
        log.exception("decision for %s failed", thread_id)
        raise HTTPException(500, "Could not record the decision. See server logs.")
    finally:
        _release(request, thread_id)


@app.post("/assessments/{thread_id}/qna")
async def qna(thread_id: str, body: QnaRequest, request: Request):
    rt = _rt(request)
    report_md = await chat.load_report(rt, thread_id)
    if not report_md:
        raise HTTPException(404, "No report found for this thread_id.")
    history = [(t.role, t.content) for t in body.history]
    try:
        answer = await chat.answer_question(report_md, history, body.question.strip())
    except Exception:
        log.exception("qna for %s failed", thread_id)
        raise HTTPException(502, "The answer service is unavailable right now. Please retry.")
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    on_windows = sys.platform == "win32"
    server = uvicorn.Server(uvicorn.Config(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        loop="none" if on_windows else "auto",   # psycopg async needs the Selector loop on Windows
    ))
    if on_windows:
        asyncio.run(server.serve(),
                    loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
    else:
        server.run()
