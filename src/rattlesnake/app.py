import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.engine import Engine
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.status import HTTP_200_OK, HTTP_409_CONFLICT

from .db import get_async_engine, get_session
from .models import DecisionRequest, DecisionResponse, HandledMessage, MessageDecision, ReplyRequest

# ---------- request/response models (lightweight, no pydantic v2 drama) ----------


def _now_utc() -> datetime:
    return datetime.now(UTC)


# ---------- routes ----------


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True}, status_code=HTTP_200_OK)


async def ingest_message(request: Request) -> JSONResponse:
    """
    POST /v1/events/message

    Body example:
    {
      "bot_id": "nyah",
      "channel_id": 123,
      "message_id": 456,
      "author_id": 789,
      "content": "anyone know what jsonb is for",
      "reply_to_message_id": null,
      "mentions": ["nyah"],
      "timestamp": "2026-01-07T12:34:56Z"
    }

    For now, this just stores a placeholder decision (you’ll plug your router logic in).
    """
    body = await request.body()
    req = DecisionRequest.model_validate_json(body)

    # TODO: plug in actual heuristics/embeddings router here
    decision_payload: dict[str, Any] = {
        "candidates": [],  # e.g. [{"bot_id":"nyah","score":0.9}]
        "version": 1,
        "generated_at": _now_utc().isoformat(),
    }

    decision = MessageDecision(
        channel_id=req.channel_id,
        message_id=req.message_id,
        candidates=decision_payload,
        scores=None,
        reasons=None,
    )

    async with get_session() as session:
        session.add(decision)
        await session.commit()

    resp = DecisionResponse(channel_id=req.channel_id, message_id=req.message_id, decision=decision_payload)

    return JSONResponse(
        resp.model_dump(),
        status_code=HTTP_200_OK,
    )


async def get_decision(request: Request) -> JSONResponse:
    """
    GET /v1/decisions/{message_id}?channel_id=...&bot_id=...

    Returns whether bot_id is in candidates, plus the stored decision blob.
    """
    app: Starlette = request.app  # type: ignore  # ty:ignore[unused-ignore-comment]

    message_id = int(request.path_params["message_id"])
    channel_id = int(request.query_params.get("channel_id", "0"))
    bot_id = request.query_params.get("bot_id")

    if channel_id == 0:
        return JSONResponse({"error": "channel_id is required"}, status_code=400)
    if not bot_id:
        return JSONResponse({"error": "bot_id is required"}, status_code=400)

    async with get_session() as session:
        decision = session.exec(
            MessageDecision.select().where(  # ty:ignore[unresolved-attribute]
                (MessageDecision.channel_id == channel_id) & (MessageDecision.message_id == message_id)
            )
        ).first()

        handled = session.exec(
            HandledMessage.select().where(  # ty:ignore[unresolved-attribute]
                (HandledMessage.channel_id == channel_id) & (HandledMessage.message_id == message_id)
            )
        ).first()

    if decision is None:
        return JSONResponse(
            {
                "channel_id": channel_id,
                "message_id": message_id,
                "bot_id": bot_id,
                "should_respond": False,
                "reason": "no_decision_cached",
                "decision": None,
                "handled": bool(handled),
            },
            status_code=HTTP_200_OK,
        )

    candidates = decision.candidates.get("candidates", [])
    should = any(c.get("bot_id") == bot_id for c in candidates) and handled is None

    return JSONResponse(
        {
            "channel_id": channel_id,
            "message_id": message_id,
            "bot_id": bot_id,
            "should_respond": should,
            "decision": decision.candidates,
            "handled": bool(handled),
        },
        status_code=HTTP_200_OK,
    )


async def mark_handled(request: Request) -> JSONResponse:
    """
    POST /v1/events/bot_reply

    Body:
    {
      "channel_id": 123,
      "message_id": 456,
      "channel_type": "text",
      "bot_id": "nyah",
      "reply_message_id": 999
    }

    Uses unique constraint on (channel_id, message_id) as the concurrency gate.
    """

    body = await request.body()
    req = ReplyRequest.model_validate_json(body)

    async with get_session() as session:
        existing: HandledMessage | None = session.exec(
            HandledMessage.select().where(  # ty:ignore[unresolved-attribute]
                (HandledMessage.channel_id == req.channel_id) & (HandledMessage.message_id == req.message_id)
            )
        ).first()

        if existing is not None:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": "already_handled",
                    "handled_by": existing.bot_id,
                    "reply_message_id": existing.reply_message_id,
                },
                status_code=HTTP_409_CONFLICT,
            )

        session.add(
            HandledMessage(
                bot_id=req.bot_id,
                channel_id=req.channel_id,
                message_id=req.message_id,
                reply_message_id=req.reply_message_id,
            )
        )

    return JSONResponse({"ok": True}, status_code=HTTP_200_OK)


# ---------- app factory ----------


def create_app() -> Starlette:
    routes = [
        Route("/healthz", health, methods=["GET"]),
        Route("/v1/events/message", ingest_message, methods=["POST"]),
        Route("/v1/decisions/{message_id:int}", get_decision, methods=["GET"]),
        Route("/v1/events/bot_reply", mark_handled, methods=["POST"]),
    ]

    app = Starlette(debug=os.getenv("DEBUG", "0") == "1", routes=routes)

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.engine = get_async_engine()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        engine: Engine = app.state.engine
        engine.dispose()

    return app


app = create_app()
