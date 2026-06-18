"""V2 supervisor state — Phase 1 (behind ``use_v2`` flag, default OFF).

The V2 architecture replaces the switch-style graph (triage → 10 leaves → END)
with a single supervisor node that owns a native tool-calling loop. Where the
legacy ``AgentState`` (app/agent/state.py) carries ~25 flow flags, the V2 state
is deliberately FLAT: ``MessagesState`` already provides ``messages`` with the
``add_messages`` reducer, and that message history is the single source of
truth for the loop. We add only the minimal identity context the loop needs.

Identity fields — aligned with what the dossier pipeline actually consumes
(see app/agent/dossier.py::handoff_dossier_pipeline, which reads ``messages``
and ``phone_hash`` and opens its own DB session):

    phone_hash:  the customer's hashed phone (SHA256+salt). This is the field
                 the dossier persists and the lead is keyed by. In production
                 it also equals the checkpointer ``thread_id``.
    thread_id:   the LangGraph conversation key (mirrors phone_hash today).
                 Kept on the state so a future tool can reference it without
                 reading ``config``. The dossier does NOT need it.

Phase 0 had ``session_id``/``client_phone`` placeholders; they were dropped
because the dossier consumes ``phone_hash`` (not a raw phone) and nothing in
the real handoff path uses a separate session id.

This module is NEW and touches nothing in the live flow.
"""
from langgraph.graph import MessagesState


class AgentStateV2(MessagesState):
    """Flat supervisor state.

    Inherits ``messages: Annotated[list, add_messages]`` from ``MessagesState``.
    Both identity fields are optional for the smoke/replay harness, which can
    invoke the graph with only ``messages`` (the supervisor still runs; only
    a real ``escalar_humano`` call needs ``phone_hash`` populated).
    """

    phone_hash: str
    thread_id: str
