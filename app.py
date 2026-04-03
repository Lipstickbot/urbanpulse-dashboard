"""
Smart City Management Dashboard backend.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload

Local LoRA mode:
    1. Put your adapter folder on this machine.
    2. Set LOCAL_LORA_ADAPTER_PATH in .env
    3. Start the server and it will try local inference first.

Optional environment variables in .env:
    LOCAL_LORA_ADAPTER_PATH=C:/path/to/adapter
    LOCAL_BASE_MODEL=unsloth/qwen2.5-1.5b-instruct-bnb-4bit
    OPENAI_API_KEY=your_key
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_MODEL=gpt-4.1-mini
    FINE_TUNED_MODEL=
    APP_HOST=127.0.0.1
    APP_PORT=8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import csv
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

try:
    from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
except ImportError:
    APIConnectionError = APIError = APITimeoutError = RateLimitError = Exception
    OpenAI = None

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    PeftModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("urbanpulse.backend")


class CityDataInput(BaseModel):
    category: str = Field(..., min_length=2, max_length=50, examples=["Traffic", "Ecology"])
    density: float = Field(..., ge=0, le=200, description="Traffic density in percent-like scale")
    avg_speed: float = Field(..., ge=0, le=200, description="Average speed in km/h")
    aqi: float = Field(..., ge=0, le=1000, description="Air Quality Index")
    pm2_5: float = Field(..., ge=0, le=1000, description="Fine particulate matter concentration")


class AIActionResponse(BaseModel):
    what_is_happening: str = Field(..., min_length=8, max_length=240)
    criticality: Literal["High (Red Zone)", "Medium (Yellow Zone)", "Low (Green Zone)"]
    recommended_actions: list[str] = Field(..., min_length=3, max_length=3)


class AnalyzeEnvelope(BaseModel):
    input: CityDataInput
    result: AIActionResponse
    source: Literal["llm", "fallback"]
    model: str


class MetricSeries(BaseModel):
    labels: list[str]
    density: list[float]
    avg_flow: list[float]
    aqi: list[float]
    pm25: list[float]


class NotificationItem(BaseModel):
    title: str
    message: str
    level: Literal["info", "warning", "critical"]


class DashboardDataResponse(BaseModel):
    overview: MetricSeries
    traffic: MetricSeries
    ecology: MetricSeries
    notifications: list[NotificationItem]
    kpis: dict[str, dict[str, float | str]]


class AIChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    city_data: CityDataInput
    history: list[dict[str, str]] = Field(default_factory=list)


class AIChatResponse(BaseModel):
    reply: str
    criticality: Literal["High (Red Zone)", "Medium (Yellow Zone)", "Low (Green Zone)"]
    quick_actions: list[str] = Field(min_length=3, max_length=3)
    source: Literal["llm", "fallback"]
    model: str


app = FastAPI(
    title="UrbanPulse AI Incident Assistant",
    version="1.0.0",
    description="LLM-powered incident assistant for a smart city dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "").strip()
LOCAL_LORA_ADAPTER_PATH = os.getenv("LOCAL_LORA_ADAPTER_PATH", "").strip()
LOCAL_BASE_MODEL = os.getenv("LOCAL_BASE_MODEL", "unsloth/qwen2.5-1.5b-instruct-bnb-4bit").strip()
TRAFFIC_CSV_PATH = Path(os.getenv("TRAFFIC_CSV_PATH", r"C:\Users\vovot\Downloads\archive (6)\smart_city_traffic_stress_dataset.csv"))
AIR_CSV_PATH = Path(os.getenv("AIR_CSV_PATH", r"C:\Users\vovot\Downloads\archive (5)\Air_Pollution_data.csv"))
SENSOR_CSV_PATH = Path(os.getenv("SENSOR_CSV_PATH", r"C:\Users\vovot\Downloads\archive (4)\smart_city_sensor_data.csv"))

client: OpenAI | None = None
if OPENAI_API_KEY and OpenAI is not None:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:
    logger.warning("OPENAI_API_KEY is not configured. Backend will use fallback mode.")

local_tokenizer = None
local_model = None


def load_local_lora_model() -> None:
    global local_model, local_tokenizer

    if not LOCAL_LORA_ADAPTER_PATH:
        return
    if AutoTokenizer is None or AutoModelForCausalLM is None or PeftModel is None or torch is None:
        logger.warning("Local model dependencies are not installed. Skipping local LoRA load.")
        return

    adapter_path = Path(LOCAL_LORA_ADAPTER_PATH)
    if not adapter_path.exists():
        logger.warning("LOCAL_LORA_ADAPTER_PATH does not exist: %s", adapter_path)
        return

    try:
        logger.info("Loading local base model: %s", LOCAL_BASE_MODEL)
        local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_MODEL, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_BASE_MODEL,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        logger.info("Loading local LoRA adapter from: %s", adapter_path)
        local_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        local_model.eval()
    except Exception as exc:
        local_model = None
        local_tokenizer = None
        logger.warning("Local LoRA model could not be loaded yet: %s", exc)


load_local_lora_model()


SYSTEM_PROMPT = """
You are UrbanPulse, a City Manager Assistant specialized for Smart City dashboards.

Your task is to convert urban sensor readings into a strict incident response JSON.

Decision policy:
- If density > 80 or aqi > 150, this is a crisis and criticality must be "High (Red Zone)".
- If values are generally low and stable, use "Low (Green Zone)".
- Otherwise, use "Medium (Yellow Zone)" when stress is building or moderate.

Domain interpretation:
- density: road load and congestion pressure
- avg_speed: mobility quality; low speed suggests traffic stress
- aqi: air quality risk level
- pm2_5: pollution severity
- category: operating context, usually Traffic or Ecology

Output rules:
- Return only valid JSON.
- No markdown, no explanation, no extra keys.
- "what_is_happening" must be one concise operational summary.
- "recommended_actions" must contain exactly 3 short, practical city actions.
- Actions must match the category and severity.
- Be decisive and practical.
""".strip()


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "what_is_happening": {"type": "string"},
        "criticality": {
            "type": "string",
            "enum": ["High (Red Zone)", "Medium (Yellow Zone)", "Low (Green Zone)"],
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["what_is_happening", "criticality", "recommended_actions"],
}


def fallback_analysis(payload: CityDataInput) -> AIActionResponse:
    category = payload.category.lower()
    crisis = payload.density > 80 or payload.aqi > 150
    medium = (
        payload.density > 55
        or payload.avg_speed < 30
        or payload.aqi > 100
        or payload.pm2_5 > 35
    )

    if crisis:
        if "traffic" in category:
            return AIActionResponse(
                what_is_happening="Severe traffic congestion is disrupting movement across a critical corridor.",
                criticality="High (Red Zone)",
                recommended_actions=[
                    "Extend green-light timing on overloaded intersections immediately",
                    "Push rerouting alerts to drivers and transit operators",
                    "Dispatch traffic control teams to the highest-density sector",
                ],
            )
        if "eco" in category or "air" in category:
            return AIActionResponse(
                what_is_happening="Dangerous air pollution levels are creating a citywide health risk.",
                criticality="High (Red Zone)",
                recommended_actions=[
                    "Issue a public health warning for affected districts",
                    "Limit high-emission traffic in the polluted zone",
                    "Increase monitoring frequency and notify emergency response staff",
                ],
            )
        return AIActionResponse(
            what_is_happening="Multiple urban indicators have crossed crisis thresholds and need immediate action.",
            criticality="High (Red Zone)",
            recommended_actions=[
                "Activate emergency operating procedures for the impacted sector",
                "Alert field teams and public communication channels immediately",
                "Increase live monitoring while mitigation measures are deployed",
            ],
        )

    if medium:
        if "traffic" in category:
            return AIActionResponse(
                what_is_happening="Traffic pressure is rising and congestion is likely without intervention.",
                criticality="Medium (Yellow Zone)",
                recommended_actions=[
                    "Adjust signal timing to smooth traffic flow",
                    "Publish alternate-route recommendations for drivers",
                    "Monitor the hottest junctions for escalation",
                ],
            )
        if "eco" in category or "air" in category:
            return AIActionResponse(
                what_is_happening="Air quality is degraded and trending toward an unhealthy range.",
                criticality="Medium (Yellow Zone)",
                recommended_actions=[
                    "Warn sensitive groups to reduce outdoor exposure",
                    "Inspect probable emission hotspots in the area",
                    "Prepare temporary restrictions if readings continue rising",
                ],
            )
        return AIActionResponse(
            what_is_happening="City conditions show moderate strain and require preventive management.",
            criticality="Medium (Yellow Zone)",
            recommended_actions=[
                "Apply targeted preventive controls in the affected area",
                "Notify operators to watch for threshold escalation",
                "Increase sensor polling and incident review frequency",
            ],
        )

    return AIActionResponse(
        what_is_happening="City conditions are stable and operating within normal limits.",
        criticality="Low (Green Zone)",
        recommended_actions=[
            "Maintain normal monitoring across active sensors",
            "Log current readings as a baseline for trend analysis",
            "Keep response teams on routine operational readiness",
        ],
    )


def fallback_chat(request: AIChatRequest) -> AIChatResponse:
    incident = fallback_analysis(request.city_data)
    user_text = request.message.lower().strip()
    category = request.city_data.category

    if any(token in user_text for token in ["action", "do", "what now", "что делать", "что делать?", "нестеу", "help"]):
        reply = (
            f"For {category}, the current state is {incident.criticality}. "
            f"{incident.what_is_happening} First priority: {incident.recommended_actions[0]}."
        )
    elif any(token in user_text for token in ["why", "почему", "неге", "reason"]):
        reply = (
            f"The alert level is {incident.criticality} because density is {request.city_data.density}, "
            f"average speed is {request.city_data.avg_speed}, AQI is {request.city_data.aqi}, "
            f"and PM2.5 is {request.city_data.pm2_5}."
        )
    elif any(token in user_text for token in ["summary", "summarize", "коротко", "brief"]):
        reply = incident.what_is_happening
    else:
        reply = (
            f"I am monitoring the {category} stream in real time. "
            f"Current status: {incident.what_is_happening} "
            f"Recommended next move: {incident.recommended_actions[0]}"
        )

    return AIChatResponse(
        reply=reply,
        criticality=incident.criticality,
        quick_actions=incident.recommended_actions,
        source="fallback",
        model=get_active_model(),
    )


def safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def day_labels(count: int) -> list[str]:
    return [f"Day {index:02d}" for index in range(1, count + 1)]


def load_dashboard_data() -> DashboardDataResponse:
    traffic_density: list[float] = []
    traffic_speed: list[float] = []
    traffic_aqi: list[float] = []
    traffic_pm25: list[float] = []

    with TRAFFIC_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            if idx >= 12:
                break
            density = safe_float(row.get("traffic_density"))
            speed = safe_float(row.get("avg_speed"))
            stress = safe_float(row.get("stress_index"))
            traffic_density.append(density)
            traffic_speed.append(speed)
            traffic_aqi.append(round(90 + stress * 0.55, 2))
            traffic_pm25.append(round(18 + stress * 0.45, 2))

    eco_aqi: list[float] = []
    eco_pm25: list[float] = []
    eco_density: list[float] = []
    eco_speed: list[float] = []

    with AIR_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            if idx >= 12:
                break
            aqi = safe_float(row.get("air_quality_index"))
            pm25 = safe_float(row.get("pm2_5"))
            eco_aqi.append(round(aqi, 2))
            eco_pm25.append(round(pm25, 2))
            eco_density.append(35.0)
            eco_speed.append(round(max(20.0, 55 - idx), 2))

    sensor_density: list[float] = []
    sensor_speed: list[float] = []
    sensor_aqi: list[float] = []
    sensor_pm25: list[float] = []

    with SENSOR_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            if idx >= 12:
                break
            vehicle_count = safe_float(row.get("Araç Sayısı"))
            occupancy = safe_float(row.get("Doluluk Oranı"))
            noise = safe_float(row.get("Gürültü Seviyesi"))
            sensor_density.append(round(vehicle_count or occupancy, 2))
            sensor_speed.append(round(max(24.0, 46 - idx), 2))
            sensor_aqi.append(round(80 + occupancy * 0.7 + noise * 0.4, 2))
            sensor_pm25.append(round(18 + occupancy * 0.5 + noise * 0.25, 2))

    overview = MetricSeries(
        labels=day_labels(len(sensor_density)),
        density=sensor_density,
        avg_flow=sensor_speed,
        aqi=sensor_aqi,
        pm25=sensor_pm25,
    )
    traffic = MetricSeries(
        labels=day_labels(len(traffic_density)),
        density=traffic_density,
        avg_flow=traffic_speed,
        aqi=traffic_aqi,
        pm25=traffic_pm25,
    )
    ecology = MetricSeries(
        labels=day_labels(len(eco_aqi)),
        density=eco_density,
        avg_flow=eco_speed,
        aqi=eco_aqi,
        pm25=eco_pm25,
    )

    notifications = [
        NotificationItem(
            title="Traffic dataset synced",
            message=f"{len(traffic_density)} daily points loaded from traffic stress data.",
            level="info",
        ),
        NotificationItem(
            title="Ecology risk detected",
            message=f"Latest AQI reached {round(max(eco_aqi), 2) if eco_aqi else 0}, keep public advisory ready.",
            level="critical" if eco_aqi and max(eco_aqi) > 300 else "warning",
        ),
        NotificationItem(
            title="Sensor stream refreshed",
            message=f"{len(sensor_density)} smart-city sensor points are available for map analytics.",
            level="info",
        ),
    ]

    kpis = {
        "overview": {
            "density": sensor_density[-1] if sensor_density else 0.0,
            "avg_speed": sensor_speed[-1] if sensor_speed else 0.0,
            "aqi": sensor_aqi[-1] if sensor_aqi else 0.0,
            "pm2_5": sensor_pm25[-1] if sensor_pm25 else 0.0,
        },
        "traffic": {
            "density": traffic_density[-1] if traffic_density else 0.0,
            "avg_speed": traffic_speed[-1] if traffic_speed else 0.0,
            "aqi": traffic_aqi[-1] if traffic_aqi else 0.0,
            "pm2_5": traffic_pm25[-1] if traffic_pm25 else 0.0,
        },
        "ecology": {
            "density": eco_density[-1] if eco_density else 0.0,
            "avg_speed": eco_speed[-1] if eco_speed else 0.0,
            "aqi": eco_aqi[-1] if eco_aqi else 0.0,
            "pm2_5": eco_pm25[-1] if eco_pm25 else 0.0,
        },
    }

    return DashboardDataResponse(
        overview=overview,
        traffic=traffic,
        ecology=ecology,
        notifications=notifications,
        kpis=kpis,
    )


def get_active_model() -> str:
    if local_model is not None:
        return f"local-lora:{Path(LOCAL_LORA_ADAPTER_PATH).name}"
    return FINE_TUNED_MODEL or OPENAI_MODEL


def extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return match.group(0)


def analyze_with_local_model(payload: CityDataInput) -> AIActionResponse:
    if local_model is None or local_tokenizer is None:
        raise RuntimeError("Local model is not loaded")

    prompt = (
        "You are UrbanPulse, a Smart City AI Incident Assistant. "
        "Return only valid JSON with keys what_is_happening, criticality, recommended_actions. "
        f"category={payload.category}, density={payload.density}, avg_speed={payload.avg_speed}, "
        f"aqi={payload.aqi}, pm2_5={payload.pm2_5}."
    )

    messages = [{"role": "user", "content": prompt}]
    if hasattr(local_tokenizer, "apply_chat_template"):
        input_text = local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = prompt

    inputs = local_tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to(local_model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = local_model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            temperature=0.1,
            pad_token_id=local_tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = local_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    json_text = extract_json_object(generated_text)
    return AIActionResponse.model_validate_json(json_text)


def analyze_with_llm(payload: CityDataInput) -> AIActionResponse:
    if client is None:
        raise RuntimeError("LLM client is not configured")

    response = client.responses.create(
        model=get_active_model(),
        temperature=0.2,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze the following Smart City dashboard readings and return strict JSON only:\n"
                    f"{json.dumps(payload.model_dump(), ensure_ascii=False)}"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "urbanpulse_incident_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            }
        },
    )
    return AIActionResponse.model_validate_json(response.output_text)


def chat_with_llm(request: AIChatRequest) -> AIChatResponse:
    if client is None:
        raise RuntimeError("LLM client is not configured")

    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in request.history[-6:]
        if item.get("content")
    )
    context = {
        "category": request.city_data.category,
        "density": request.city_data.density,
        "avg_speed": request.city_data.avg_speed,
        "aqi": request.city_data.aqi,
        "pm2_5": request.city_data.pm2_5,
        "user_message": request.message,
        "recent_history": history_text,
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reply": {"type": "string"},
            "criticality": {
                "type": "string",
                "enum": ["High (Red Zone)", "Medium (Yellow Zone)", "Low (Green Zone)"],
            },
            "quick_actions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["reply", "criticality", "quick_actions"],
    }

    response = client.responses.create(
        model=get_active_model(),
        temperature=0.35,
        input=[
            {
                "role": "system",
                "content": (
                    "You are UrbanPulse AI, a real-time Smart City dashboard copilot. "
                    "Answer like an operations assistant in a short chat style. "
                    "Use the provided city metrics. If density > 80 or AQI > 150, criticality must be High (Red Zone). "
                    "Return strict JSON only with reply, criticality, quick_actions. "
                    "Keep reply under 90 words and make it practical."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "urbanpulse_chat_response",
                "strict": True,
                "schema": schema,
            }
        },
    )
    data = json.loads(response.output_text)
    return AIChatResponse(
        reply=data["reply"],
        criticality=data["criticality"],
        quick_actions=data["quick_actions"],
        source="llm",
        model=get_active_model(),
    )


@app.get("/", response_class=FileResponse)
def dashboard() -> FileResponse:
    return FileResponse(BASE_DIR / "dashboard.html")


@app.get("/health")
def healthcheck() -> dict[str, object]:
    return {
        "status": "ok",
        "llm_configured": client is not None,
        "active_model": get_active_model(),
    }


@app.get("/dashboard-data", response_model=DashboardDataResponse)
def dashboard_data() -> DashboardDataResponse:
    return load_dashboard_data()


@app.post("/analyze", response_model=AnalyzeEnvelope)
def analyze(payload: CityDataInput) -> AnalyzeEnvelope:
    model_name = get_active_model()
    try:
        result = analyze_with_local_model(payload) if local_model is not None else analyze_with_llm(payload)
        return AnalyzeEnvelope(input=payload, result=result, source="llm", model=model_name)
    except (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
        ValidationError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
    ) as exc:
        logger.warning("LLM failed, fallback activated: %s", exc)
        result = fallback_analysis(payload)
        return AnalyzeEnvelope(input=payload, result=result, source="fallback", model=model_name)


@app.post("/chat", response_model=AIChatResponse)
def chat(request: AIChatRequest) -> AIChatResponse:
    try:
        return chat_with_llm(request)
    except (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError,
        ValidationError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
    ) as exc:
        logger.warning("Chat LLM failed, fallback activated: %s", exc)
        return fallback_chat(request)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)
