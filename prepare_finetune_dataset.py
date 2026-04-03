"""
Prepare instruction-style JSONL data for fine-tuning or supervised adaptation.

Run:
    python prepare_finetune_dataset.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "training_data"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAFFIC_CSV = Path(r"C:\Users\vovot\Downloads\archive (6)\smart_city_traffic_stress_dataset.csv")
AIR_CSV = Path(r"C:\Users\vovot\Downloads\archive (5)\Air_Pollution_data.csv")
SENSOR_CSV = Path(r"C:\Users\vovot\Downloads\archive (4)\smart_city_sensor_data.csv")


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def classify(category: str, density: float, avg_speed: float, aqi: float, pm2_5: float) -> dict[str, object]:
    category_lower = category.lower()
    if density > 80 or aqi > 150:
        criticality = "High (Red Zone)"
        if "traffic" in category_lower:
            text = "Critical congestion is causing severe mobility disruption."
            actions = [
                "Extend green-light timing on saturated intersections",
                "Send rerouting alerts to drivers and transit services",
                "Deploy traffic teams to the most congested corridors",
            ]
        else:
            text = "Hazardous air conditions are creating an immediate public health risk."
            actions = [
                "Issue an air-quality alert for affected neighborhoods",
                "Restrict high-emission mobility in the impacted zone",
                "Increase monitoring and notify city response teams",
            ]
    elif density > 55 or avg_speed < 30 or aqi > 100 or pm2_5 > 35:
        criticality = "Medium (Yellow Zone)"
        if "traffic" in category_lower:
            text = "Traffic pressure is building and may escalate soon."
            actions = [
                "Adjust traffic-signal phases for smoother flow",
                "Publish alternate route guidance for drivers",
                "Track hotspot intersections for escalation",
            ]
        else:
            text = "Air quality is degraded and requires preventive action."
            actions = [
                "Warn sensitive groups to limit exposure",
                "Inspect likely pollution sources in the area",
                "Prepare escalation controls if readings worsen",
            ]
    else:
        criticality = "Low (Green Zone)"
        if "traffic" in category_lower:
            text = "Traffic conditions remain stable and manageable."
            actions = [
                "Maintain standard signal plans",
                "Continue routine traffic monitoring",
                "Log readings for trend analysis",
            ]
        else:
            text = "Environmental conditions are currently within normal limits."
            actions = [
                "Maintain regular air-quality monitoring",
                "Record baseline environmental readings",
                "Keep routine city operations unchanged",
            ]

    return {
        "what_is_happening": text,
        "criticality": criticality,
        "recommended_actions": actions,
    }


def build_example(category: str, density: float, avg_speed: float, aqi: float, pm2_5: float) -> dict[str, object]:
    prompt = {
        "category": category,
        "density": round(density, 2),
        "avg_speed": round(avg_speed, 2),
        "aqi": round(aqi, 2),
        "pm2_5": round(pm2_5, 2),
    }
    completion = classify(category, density, avg_speed, aqi, pm2_5)
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are UrbanPulse, a City Manager Assistant. "
                    "Return only JSON with what_is_happening, criticality, and recommended_actions."
                ),
            },
            {
                "role": "user",
                "content": f"Analyze this city data and respond with strict JSON only: {json.dumps(prompt)}",
            },
            {
                "role": "assistant",
                "content": json.dumps(completion, ensure_ascii=False),
            },
        ]
    }


def traffic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with TRAFFIC_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                build_example(
                    category="Traffic",
                    density=safe_float(row.get("traffic_density")),
                    avg_speed=safe_float(row.get("avg_speed")),
                    aqi=40.0,
                    pm2_5=18.0,
                )
            )
    return rows


def air_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with AIR_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                build_example(
                    category="Ecology",
                    density=35.0,
                    avg_speed=50.0,
                    aqi=safe_float(row.get("air_quality_index")),
                    pm2_5=safe_float(row.get("pm2_5")),
                )
            )
    return rows


def sensor_rows(limit: int = 300) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SENSOR_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            vehicle_count = safe_float(row.get("Araç Sayısı"))
            noise = safe_float(row.get("Gürültü Seviyesi"))
            occupancy = safe_float(row.get("Doluluk Oranı"))
            sensor_type = (row.get("Sensör Tipi") or "").lower()

            if "trafik" in sensor_type:
                category = "Traffic"
                density = min(vehicle_count, 120.0)
                avg_speed = max(12.0, 70.0 - (density * 0.4))
                aqi = 60.0 + (noise * 0.2)
                pm2_5 = 20.0 + (noise * 0.1)
            else:
                category = "Ecology"
                density = min(occupancy, 100.0)
                avg_speed = 45.0
                aqi = 50.0 + (noise * 0.3) + (occupancy * 0.4)
                pm2_5 = 15.0 + (occupancy * 0.5)

            rows.append(build_example(category, density, avg_speed, aqi, pm2_5))
    return rows


def main() -> None:
    dataset = traffic_rows() + air_rows() + sensor_rows()
    output_path = OUTPUT_DIR / "urbanpulse_finetune.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in dataset:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} training examples to {output_path}")


if __name__ == "__main__":
    main()
