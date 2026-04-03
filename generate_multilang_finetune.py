"""
Generate a multilingual Smart City instruction dataset for QLoRA fine-tuning.

Run:
    python generate_multilang_finetune.py

Output:
    training_data/urbanpulse_multilang_qlora.jsonl
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "training_data"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAFFIC_CSV = Path(r"C:\Users\vovot\Downloads\archive (6)\smart_city_traffic_stress_dataset.csv")
AIR_CSV = Path(r"C:\Users\vovot\Downloads\archive (5)\Air_Pollution_data.csv")
SENSOR_CSV = Path(r"C:\Users\vovot\Downloads\archive (4)\smart_city_sensor_data.csv")
OUTPUT_FILE = OUTPUT_DIR / "urbanpulse_multilang_qlora.jsonl"


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def classify(density: float, avg_speed: float, aqi: float, pm2_5: float) -> tuple[str, list[str]]:
    if density > 80 or aqi > 150:
        return (
            "High (Red Zone)",
            [
                "Extend green light phases on overloaded intersections",
                "Send emergency alerts and rerouting notifications",
                "Dispatch city response teams to the affected zone",
            ],
        )
    if density > 55 or avg_speed < 30 or aqi > 100 or pm2_5 > 35:
        return (
            "Medium (Yellow Zone)",
            [
                "Adjust traffic and monitoring settings preventively",
                "Warn operators about a developing incident",
                "Increase sensor observation frequency",
            ],
        )
    return (
        "Low (Green Zone)",
        [
            "Maintain routine monitoring",
            "Log current values as baseline conditions",
            "Keep normal city operations active",
        ],
    )


def build_summary(category: str, criticality: str, density: float, avg_speed: float, aqi: float, pm2_5: float, lang: str) -> str:
    if lang == "ru":
        if criticality == "High (Red Zone)":
            return f"Обнаружен критический инцидент в категории {category}: высокая нагрузка на городскую систему, density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."
        if criticality == "Medium (Yellow Zone)":
            return f"Обнаружено повышенное напряжение в категории {category}: ситуация требует превентивных мер, density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."
        return f"Ситуация в категории {category} стабильна: показатели в допустимом диапазоне, density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."

    if lang == "kz":
        if criticality == "High (Red Zone)":
            return f"{category} санатында қауіпті деңгейдегі оқиға анықталды: density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."
        if criticality == "Medium (Yellow Zone)":
            return f"{category} санатында тәуекел өсіп жатыр: алдын алу шаралары қажет, density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."
        return f"{category} санатындағы жағдай тұрақты: көрсеткіштер қалыпты деңгейде, density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."

    if criticality == "High (Red Zone)":
        return f"A critical {category} incident is underway with dangerous readings: density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."
    if criticality == "Medium (Yellow Zone)":
        return f"A moderate {category} incident is developing and requires preventive action: density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."
    return f"{category} conditions are stable with normal operating values: density={density:.1f}, speed={avg_speed:.1f}, AQI={aqi:.1f}, PM2.5={pm2_5:.1f}."


def translate_actions(actions: list[str], lang: str) -> list[str]:
    mapping = {
        "ru": [
            "Увеличить фазу зеленого сигнала на перегруженных перекрестках",
            "Отправить экстренные оповещения и маршруты объезда",
            "Направить городские службы в проблемную зону",
            "Скорректировать сигналы и мониторинг превентивно",
            "Предупредить операторов о нарастающем инциденте",
            "Увеличить частоту наблюдения сенсоров",
            "Сохранить стандартный режим мониторинга",
            "Зафиксировать текущие показатели как базовую линию",
            "Сохранить штатный режим городских систем",
        ],
        "kz": [
            "Жүктелген қиылыстарда жасыл шам уақытын ұзарту",
            "Шұғыл ескертулер мен айналма маршруттарды жіберу",
            "Қалалық жедел топтарды мәселелі аймаққа жіберу",
            "Бағдаршамдар мен мониторингті алдын ала реттеу",
            "Операторларға өсіп келе жатқан оқиға туралы ескерту",
            "Сенсорларды бақылау жиілігін арттыру",
            "Қалыпты мониторинг режимін сақтау",
            "Ағымдағы көрсеткіштерді базалық деңгей ретінде тіркеу",
            "Қалалық жүйелердің штаттық режимін сақтау",
        ],
    }
    if lang == "en":
        return actions

    groups = {
        "Extend green light phases on overloaded intersections": 0,
        "Send emergency alerts and rerouting notifications": 1,
        "Dispatch city response teams to the affected zone": 2,
        "Adjust traffic and monitoring settings preventively": 3,
        "Warn operators about a developing incident": 4,
        "Increase sensor observation frequency": 5,
        "Maintain routine monitoring": 6,
        "Log current values as baseline conditions": 7,
        "Keep normal city operations active": 8,
    }
    translated = mapping[lang]
    return [translated[groups[action]] for action in actions]


def build_prompt(category: str, density: float, avg_speed: float, aqi: float, pm2_5: float, lang: str) -> str:
    if lang == "ru":
        return (
            "Ты AI City Manager Assistant. Проанализируй показатели умного города и верни только JSON "
            f"с полями what_is_happening, criticality, recommended_actions. category={category}, "
            f"density={density:.1f}, avg_speed={avg_speed:.1f}, aqi={aqi:.1f}, pm2_5={pm2_5:.1f}."
        )
    if lang == "kz":
        return (
            "Сен Smart City AI Assistant моделісің. Көрсеткіштерді талдап, тек JSON қайтар: "
            f"what_is_happening, criticality, recommended_actions. category={category}, "
            f"density={density:.1f}, avg_speed={avg_speed:.1f}, aqi={aqi:.1f}, pm2_5={pm2_5:.1f}."
        )
    return (
        "You are a Smart City AI Incident Assistant. Analyze the readings and return only JSON with "
        f"what_is_happening, criticality, recommended_actions. category={category}, "
        f"density={density:.1f}, avg_speed={avg_speed:.1f}, aqi={aqi:.1f}, pm2_5={pm2_5:.1f}."
    )


def build_record(category: str, density: float, avg_speed: float, aqi: float, pm2_5: float, lang: str) -> dict[str, object]:
    criticality, actions = classify(density, avg_speed, aqi, pm2_5)
    completion = {
        "what_is_happening": build_summary(category, criticality, density, avg_speed, aqi, pm2_5, lang),
        "criticality": criticality,
        "recommended_actions": translate_actions(actions, lang),
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON for Smart City incident analysis.",
            },
            {
                "role": "user",
                "content": build_prompt(category, density, avg_speed, aqi, pm2_5, lang),
            },
            {
                "role": "assistant",
                "content": json.dumps(completion, ensure_ascii=False),
            },
        ]
    }


def collect_examples() -> list[tuple[str, float, float, float, float]]:
    examples: list[tuple[str, float, float, float, float]] = []

    with TRAFFIC_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            examples.append((
                "Traffic",
                safe_float(row.get("traffic_density")),
                safe_float(row.get("avg_speed")),
                50.0,
                18.0,
            ))

    with AIR_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            examples.append((
                "Ecology",
                35.0,
                50.0,
                safe_float(row.get("air_quality_index")),
                safe_float(row.get("pm2_5")),
            ))

    with SENSOR_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= 400:
                break
            vehicle_count = safe_float(row.get("Araç Sayısı"))
            occupancy = safe_float(row.get("Doluluk Oranı"))
            noise = safe_float(row.get("Gürültü Seviyesi"))
            sensor_type = (row.get("Sensör Tipi") or "").lower()

            if "trafik" in sensor_type:
                examples.append((
                    "Traffic",
                    min(vehicle_count, 120.0),
                    max(12.0, 70.0 - vehicle_count * 0.35),
                    55.0 + noise * 0.2,
                    20.0 + noise * 0.1,
                ))
            else:
                examples.append((
                    "Ecology",
                    min(occupancy, 100.0),
                    45.0,
                    45.0 + occupancy * 0.5 + noise * 0.2,
                    15.0 + occupancy * 0.4,
                ))

    return examples


def main() -> None:
    random.seed(42)
    base_examples = collect_examples()
    random.shuffle(base_examples)
    base_examples = base_examples[:180]

    records: list[dict[str, object]] = []
    for category, density, avg_speed, aqi, pm2_5 in base_examples:
        for lang in ("ru", "kz", "en"):
            records.append(build_record(category, density, avg_speed, aqi, pm2_5, lang))

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} multilingual records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
