# tools/get_weather.py  (fixed)
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests


class WeatherInput(BaseModel):
    """Arguments for the get_weather tool."""
    city: str = Field(..., description="City name to get the current weather for.")
    unit: str = Field(
        "metric",
        description="Temperature unit: 'metric' for Celsius or 'imperial' for Fahrenheit.",
    )


@tool("get_weather", args_schema=WeatherInput, return_direct=False)
def get_weather(city: str, unit: str = "metric") -> str:
    """
    Minimal weather tool using Open-Meteo API.
    Returns a simple weather summary for the specified city.
    """
    try:
        # Step 1: geocode
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        try:
            geo.raise_for_status()
        except Exception:
            return f"Weather tool error (geocoding): {geo.status_code} {geo.text}"

        results = (geo.json() or {}).get("results") or []
        if not results:
            return f"Weather error: city '{city}' not found."

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        name = results[0].get("name", city)

        # Step 2: current weather
        temp_unit = "fahrenheit" if unit == "imperial" else "celsius"
        wind_unit_api = "mph" if unit == "imperial" else "ms"
        wind_unit_display = "mph" if unit == "imperial" else "m/s"

        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weathercode,windspeed_10m",
                "temperature_unit": temp_unit,
                "windspeed_unit": wind_unit_api,
            },
            timeout=8,
        )
        try:
            resp.raise_for_status()
        except Exception:
            return f"Weather tool error (forecast): {resp.status_code} {resp.text}"

        current = (resp.json() or {}).get("current") or {}
        if not current:
            return f"Weather error: no current data returned for '{city}'."

        temp = current.get("temperature_2m")
        wind = current.get("windspeed_10m")
        code = current.get("weathercode")

        if temp is None or wind is None or code is None:
            return "Weather error: missing fields in API response."

        return (
            f"Weather in {name}: {temp}°{'F' if unit == 'imperial' else 'C'}, "
            f"wind {wind} {wind_unit_display}, "
            f"condition code {code}."
        )

    except Exception as e:
        return f"Weather tool error: {e}"
