"""Lightweight wrappers around Langfuse v4 observations for the MEP pipeline.

All helpers accept ``None`` as the client/trace and become no-ops, so the
rest of the codebase can call them unconditionally.
"""

import contextlib
from contextlib import contextmanager
from typing import Optional


try:
    from langfuse import propagate_attributes  # requires langfuse>=4
except Exception:

    @contextmanager  # type: ignore[misc]
    def propagate_attributes(**_: object):  # type: ignore[misc]
        """Fallback no-op context manager if langfuse v4 is not available."""
        yield


def _normalize_usage(usage: dict) -> dict:
    """
    Map provider usage dicts (OpenAI/Gemini) to Langfuse v4 usage_details keys.

    Parameters
    ----------
    usage : dict
        The raw usage dict from the provider.

    Returns
    -------
    dict
        Normalized usage details for Langfuse.
    """
    normalized: dict = {}
    # OpenAI keys
    if "prompt_tokens" in usage:
        normalized["input"] = usage["prompt_tokens"]
    elif "input" in usage:
        normalized["input"] = usage["input"]
    if "completion_tokens" in usage:
        normalized["output"] = usage["completion_tokens"]
    elif "output" in usage:
        normalized["output"] = usage["output"]
    if "total_tokens" in usage:
        normalized["total"] = usage["total_tokens"]
    elif "total" in usage:
        normalized["total"] = usage["total"]
    return normalized or usage


class _TraceHandle:
    """Thin wrapper yielded by sample_trace; exposes a stable interface across callers.

    Attributes
    ----------
    id : str | None
        The Langfuse trace ID, usable for attaching scores after the run.
    """

    def __init__(self, span: object, trace_id: Optional[str]) -> None:
        self._span = span
        self.id = trace_id

    def update(self, **kwargs: object) -> None:
        """Update the root trace span (e.g. set output after the run)."""
        if self._span is not None:
            with contextlib.suppress(Exception):
                self._span.update(**kwargs)  # type: ignore[union-attr]

    def score_trace(self, name: str, value: float) -> None:
        """Attach a numeric score to the root trace."""
        if self._span is not None:
            with contextlib.suppress(Exception):
                self._span.score_trace(name=name, value=value)  # type: ignore[union-attr]


@contextmanager
def sample_trace(
    client: object,
    sample_id: str,
    question: str,
    pre_text: str,
    table: str,
    post_text: str,
    expected_answer: str,
    config_name: str,
    run_id: str,
    project_name: str = "FinQA_eval",
):  # type: ignore[return]
    """
    Context manager to create a Langfuse trace for a single sample.

    Parameters
    ----------
    client : object
        The Langfuse client. If None, the context manager yields None.
    sample_id : str
        Unique identifier for the sample.
    question : str
        The input prompt text.
    pre_text : str
        The pre-text for the question.
    table : str
        The table associated with the question.
    post_text : str
        The post-text for the question.
    expected_answer : str
        The expected answer for the question.
    config_name : str
        The evaluation configuration used.
    run_id : str
        The unique ID of the pipeline run.
    project_name : str, default 'chartqapro-eval'
        Langfuse project identifier (ignored in v4; use SDK config).

    Yields
    ------
    trace_handle : _TraceHandle or None
        The initialized trace object.
    """
    del project_name  # kept for API compatibility; Langfuse v4 uses project from SDK config
    if client is None:
        yield None
        return

    with (
        client.start_as_current_observation(  # type: ignore[union-attr]
            name=f"chartqapro/{sample_id}",
            as_type="span",
            input={"question": question, "pre_text": pre_text, "table": table, "post_text": post_text, "expected_answer": expected_answer},
            metadata={
                "run_id": run_id,
                "config": config_name
            },
        ) as span,
        propagate_attributes(session_id=run_id),
    ):
        trace_id = client.get_current_trace_id()  # type: ignore[union-attr]
        handle = _TraceHandle(span=span, trace_id=trace_id)
        try:
            yield handle
        finally:
            with contextlib.suppress(Exception):
                client.flush()  # type: ignore[union-attr]


def open_llm_span(
    trace: object,
    name: str,
    input_data: dict,
    model: str,
    metadata: Optional[dict] = None,
    parent_span_id: Optional[str] = None,
) -> object:
    """
    Begin a Langfuse generation observation on the given trace span.

    ``parent_span_id`` is accepted for API compatibility but is unused in v4 —
    nesting is handled by calling ``start_observation`` on the parent span.

    Parameters
    ----------
    trace : object
        The parent trace or span.
    name : str
        Logical name for the operation.
    input_data : dict
        Model inputs.
    model : str
        Model identifier.
    metadata : dict, optional
        Additional context keys.
    parent_span_id : str, optional
        Explicit parent linkage (ignored in v4; nesting is contextual).

    Returns
    -------
    object or None
        The active span object.
    """
    del parent_span_id  # kept for API compatibility; v4 uses contextual nesting
    if trace is None:
        return None
    span = getattr(trace, "_span", None)
    if span is None:
        return None
    with contextlib.suppress(Exception):
        return span.start_observation(  # type: ignore[union-attr]
            name=name,
            as_type="generation",
            input=input_data,
            model=model,
            metadata=metadata or {},
        )
    return None


def close_span(
    span: object,
    output: Optional[dict] = None,
    usage: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """Log results and terminate an active span.

    Parameters
    ----------
    span : object
        The span to close.
    output : dict, optional
        The model output to log.
    usage : dict, optional
        The provider usage dict (e.g. OpenAI or Gemini keys).
    error : str, optional
        An error message to log (if any).

    Returns
    -------
    None
    """
    if span is None:
        return
    with contextlib.suppress(Exception):
        update_kwargs: dict = {}
        if output is not None:
            update_kwargs["output"] = output
        if usage:
            update_kwargs["usage_details"] = _normalize_usage(usage)
        if error:
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = error
        if update_kwargs:
            span.update(**update_kwargs)  # type: ignore[union-attr]
        span.end()  # type: ignore[union-attr]


def log_trace_scores(trace: object, scores: dict) -> None:
    """
    Attach quantitative feedback scores to a trace.

    Parameters
    ----------
    trace : object
        The trace to update.
    scores : dict
        Mapping of metric names to numeric values.

    Returns
    -------
    None
    """
    if trace is None:
        return
    for name, value in scores.items():
        if isinstance(value, (int, float)):
            with contextlib.suppress(Exception):
                if hasattr(trace, "score_trace"):
                    trace.score_trace(name=name, value=float(value))  # type: ignore[union-attr]


def open_tool_call_span(
    trace: object,
    tool_name: str,
    input_data: dict,
    metadata: Optional[dict] = None,
) -> object:
    """
    Begin a Langfuse span observation for a tool call.

    Parameters
    ----------
    trace : object
        The parent trace or span.
    tool_name : str
        Name of the tool being called (e.g., "math_ops_tool").
    input_data : dict
        The input arguments to the tool.
    metadata : dict, optional
        Additional context keys (e.g., tool version, config).

    Returns
    -------
    object or None
        The active span object for the tool call.
    """
    if trace is None:
        return None
    span = getattr(trace, "_span", None)
    if span is None:
        return None
    with contextlib.suppress(Exception):
        return span.start_observation(  # type: ignore[union-attr]
            name=tool_name,
            as_type="span",
            input=input_data,
            metadata=metadata or {},
        )
    return None


def close_tool_call_span(
    span: object,
    output: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log tool execution results and terminate the tool call span.

    Parameters
    ----------
    span : object
        The tool call span to close.
    output : dict, optional
        The tool output to log.
    error : str, optional
        An error message if the tool call failed.

    Returns
    -------
    None
    """
    if span is None:
        return
    with contextlib.suppress(Exception):
        update_kwargs: dict = {}
        if output is not None:
            update_kwargs["output"] = output
        if error:
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = error
        if update_kwargs:
            span.update(**update_kwargs)  # type: ignore[union-attr]
        span.end()  # type: ignore[union-attr]
