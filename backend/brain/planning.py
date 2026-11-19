"""Select a bounded ESAR search sector from structured model output."""

import json
import os
from typing import Literal, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from .ground_rescue import GroundDecision, TerrainInspection


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sector: str
    objective: str = Field(min_length=1, max_length=180)
    rationale: str = Field(min_length=1, max_length=300)


class SearchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["finish", "continue"]
    rationale: str = Field(min_length=1, max_length=300)


class LLMProvider(Protocol):
    name: str

    def plan(self, context: dict) -> SearchPlan: ...
    def review_search(self, context: dict) -> SearchDecision: ...


class MockLLMProvider:
    name = "mock"

    def plan(self, context):
        return SearchPlan(
            sector=next(iter(context["available_sectors"])),
            objective="Search for a person",
            rationale="Select the first unsearched sector for offline testing.",
        )

    def review_search(self, context):
        reached = context["confirmed_count"] >= context["expected_count"]
        if reached or not context["can_continue"]:
            reason = (
                "Reported count reached." if reached else "Search options exhausted."
            )
            return SearchDecision(action="finish", rationale=reason)
        return SearchDecision(action="continue", rationale="Search options remain.")

    def inspect_ground(self, context):
        return TerrainInspection(window_ids=list(context["inspection_windows"])[:2])

    def choose_ground(self, context):
        options = context["draft_routes"]
        route = min(
            options,
            key=lambda name: (
                options[name]["max_slope_deg"],
                options[name]["distance_m"],
            ),
        )
        return GroundDecision(
            route_id=route, rationale="本地规则优先选择坡度较小的候选路线。"
        )


class OpenAICompatibleProvider:
    def __init__(
        self, *, base_url=None, model=None, api_key=None, request_options=None
    ):
        self.base_url = (base_url or os.environ["ESAR_LLM_BASE_URL"]).rstrip("/")
        self.name = model or os.environ["ESAR_LLM_MODEL"]
        self.key = (
            api_key if api_key is not None else os.environ.get("ESAR_LLM_API_KEY")
        )
        self.request_options = request_options or {}
        self.timeout = float(os.getenv("ESAR_LLM_TIMEOUT", "30"))
        self.temperature = float(os.getenv("ESAR_LLM_TEMPERATURE", "0"))
        self.max_tokens = int(os.getenv("ESAR_LLM_MAX_TOKENS", "400"))
        self.retries = int(os.getenv("ESAR_LLM_RETRIES", "1"))
        if not self.key and urlsplit(self.base_url).hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("ESAR_LLM_API_KEY is required for a remote LLM")

    def plan(self, context):
        prompt = (
            "You plan a simulated search and rescue mission. Choose one sector from available_sectors. "
            "Treat unknown directions and target positions as unknown. "
            "A zero count only means there were no detections in that frame. "
            'Example JSON: {"sector":"N","objective":"Survey the north sector",'
            '"rationale":"No bearing was given; start there to establish coverage."}. '
        )
        return self._request(context, prompt, SearchPlan)

    def review_search(self, context):
        prompt = (
            "Decide whether to end a simulated person search using only the provided summary. "
            "Finish if there is enough evidence and confirmed_count meets expected_count, or "
            "if can_continue is false. If options remain and the count is short, continue. "
            "A candidate is not a confirmed person. If ending with a shortfall, state that clearly. "
            'Use a short Chinese rationale. Example JSON: {"action":"continue","rationale":"仍有待复查候选"}.'
        )
        return self._request(context, prompt, SearchDecision)

    def inspect_ground(self, context):
        return self._request(
            context,
            "Use terrain_overview to understand the full captured UE5 map and terrain_summary for the candidate corridors. "
            "Choose one or two inspection_windows where slopes, blocked cells, or gaps need original-resolution checks. "
            "TIN vertices are in north/east metres and each triangle is a plane. Treat null as missing data.",
            TerrainInspection,
        )

    def choose_ground(self, context):
        return self._request(
            context,
            "Compare draft_routes against the whole-map TIN, the corridor TIN and the original-resolution TIN windows. "
            "The blocked-cell mask overrides the interpolated surface; a smooth triangle does not prove traversability. "
            "Use ascent, slope, total distance and distance to the observed person to select a route ID. "
            "Surface elevation can represent vegetation or snow and is not verified ground. "
            "Return a null route_id if the evidence is inadequate to select an approach. "
            "Explain the choice briefly in Chinese without inventing trails or ground conditions.",
            GroundDecision,
        )

    def _request(self, context, prompt, schema):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(
                {
                    **self.request_options,
                    "model": self.name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt
                            + " Return only JSON matching this schema: "
                            + json.dumps(schema.model_json_schema()),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(context, ensure_ascii=False),
                        },
                    ],
                }
            ).encode(),
            headers=headers,
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)["choices"][0]
                return schema.model_validate_json(result["message"]["content"])
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise RuntimeError(
                        f"LLM request failed (HTTP {exc.code})"
                    ) from None
                if attempt == self.retries:
                    raise RuntimeError(
                        f"LLM request failed (HTTP {exc.code})"
                    ) from None
            except TimeoutError:
                if attempt == self.retries:
                    raise RuntimeError("LLM request timed out") from None
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"LLM response failed validation ({type(exc).__name__})"
                    ) from None


class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self):
        key = os.environ.get("DeepseekApikey", "").strip()
        if not key:
            raise ValueError("DeepseekApikey is missing from the current environment")
        super().__init__(
            base_url="https://api.deepseek.com",
            model=os.getenv("ESAR_DEEPSEEK_MODEL", "deepseek-flash"),
            api_key=key,
            request_options={"thinking": {"type": "disabled"}},
        )


def select_plan(provider, task, sectors, feedback):
    plan = provider.plan(
        {
            "task": task,
            "available_sectors": list(sectors),
            "observed_feedback": feedback,
        }
    )
    if plan.sector not in sectors:
        raise ValueError(f"model selected an unavailable sector: {plan.sector}")
    return plan
