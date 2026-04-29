"""math operations tool — performs addition, subtraction, multiplication and division

The tool takes two values and returns the math operation result.
Trace metadata is stored in a private attribute and accessible
via `tool.pop_traces()` after the Crew finishes.
"""

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr

from ..langfuse_integration.tracing import close_span, open_llm_span


class MathOpsInput(BaseModel):
    """Input schema for MathOpsTool."""

    a: float = Field(description="First numerical value for math operation")
    b: float = Field(description="Second numerical value for math operation")
    math_ops: str = Field(description="Math operation to perform on the two given numerical values of model's choice \
                          (addition | subtraction | multiplication | division)")


class MathOpsTool(BaseTool):
    """Take 2 values and perform the corresponding math operation. """

    name: str = "math_ops_tool"
    description: str = (
        "Perform math operation on two numerical values. "
        "Provide the two numerical values and the math operation of model's choice. "
        "Returns a numerical value as the result of the math operation. "
    )
    args_schema: Type[BaseModel] = MathOpsInput

    # Public config fields
    lf_trace: Optional[Any] = None  # Langfuse Trace object for span creation

    # Private mutable trace storage (not a Pydantic field)
    _traces: list = PrivateAttr(default_factory=list)

    def pop_traces(self) -> list:
        """
        Retrieve and clear the accumulated execution traces.

        Returns
        -------
        list
            A list of dictionary traces documenting each tool call.
        """
        traces = list(self._traces)
        self._traces.clear()
        return traces

    # ------------------------------------------------------------------
    # CrewAI entry point
    # ------------------------------------------------------------------

    def _run(
        self,
        a: float,
        b: float,
        math_ops: str,
    ) -> float:
        """
        Execute math operation logic.

        Parameters
        ----------
        a: float
            The first numerical value
        b: float
            The second numerical value
        math_ops: str
            The math operation
        
        Returns
        -------
        float
            The result of the math operation
        """
        start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        provider_meta: dict = {}
        error_str: Optional[str] = None
        try:
            if math_ops == "addition":
                raw_text, provider_meta = a + b, {"a": a, "b": b, "math_ops": math_ops}
            elif math_ops == "subtraction":
                raw_text, provider_meta = a - b, {"a": a, "b": b, "math_ops": math_ops}
            elif math_ops == "multiplication":
                raw_text, provider_meta = a * b, {"a": a, "b": b, "math_ops": math_ops}
            elif math_ops == "division":
                raw_text, provider_meta = a / b, {"a": a, "b": b, "math_ops": math_ops}
            else:
                raise ValueError(f"Unknown Math Operation: {self.math_ops!r}")
            raw_text = round(raw_text, 3)
        except Exception as exc:
            raw_text = json.dumps({"answer": "ERROR", "explanation": f"Tool error: {exc}"})
            provider_meta = {"error": str(exc)}
            error_str = str(exc)

        provider_meta["result"] = raw_text

        end_ts = datetime.now(timezone.utc).isoformat()
        elapsed_ms = (time.time() - t0) * 1000.0

        self._traces.append(
            {
                "tool": "math_ops_tool",
                "start_ts": start_ts,
                "end_ts": end_ts,
                "elapsed_ms": elapsed_ms,
                "provider_metadata": provider_meta,
            }
        )

        return raw_text
