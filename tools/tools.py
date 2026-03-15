"""
Ecolab Agent — Tool implementations.
Shared by both the MCP server (server.py) and the local Python agent (agent.py).
All functions are plain async — no MCP types, no framework dependency.
"""

import os
import json
import logging

import httpx
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.embedded import EmbeddedOptions

logger = logging.getLogger("ecolab.tools")

COLLECTION_NAME = "EcolabDocs"
WEAVIATE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".weaviate_data")

# ── Shared singletons ────────────────────────────────────────────────────────

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (multi-qa-MiniLM-L6-cos-v1)...")
        _embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        logger.info("Embedding model ready.")
    return _embedder


def get_weaviate_client():
    import subprocess
    os.makedirs(WEAVIATE_DATA_DIR, exist_ok=True)
    options = EmbeddedOptions(
        persistence_data_path=WEAVIATE_DATA_DIR,
        port=8079,
        grpc_port=50052,
        additional_env_vars={
            "ENABLE_MODULES": "",
            "DEFAULT_VECTORIZER_MODULE": "none",
            "LOG_LEVEL": "panic",          # only fatal errors from Weaviate process
        },
    )
    # Suppress the Weaviate subprocess stdout/stderr
    import weaviate.embedded as _we
    _orig = _we.EmbeddedDB._launch if hasattr(_we, "EmbeddedDB") else None

    client = weaviate.WeaviateClient(
        embedded_options=options,
        additional_config=weaviate.classes.init.AdditionalConfig(
            timeout=weaviate.classes.init.Timeout(init=30)
        ),
    )
    # Redirect the embedded process output after it starts
    client.connect()
    try:
        proc = client._connection._embedded_db.process  # type: ignore
        if proc and proc.stdout:
            proc.stdout = open(os.devnull, "w")
        if proc and proc.stderr:
            proc.stderr = open(os.devnull, "w")
    except Exception:
        pass
    return client


# ── Tool: RAG search ─────────────────────────────────────────────────────────

async def search_environmental_docs(query: str, top_k: int = 5) -> str:
    """Semantic search over the indexed environmental knowledge base."""
    try:
        model = get_embedder()
        query_vector = model.encode([query])[0].tolist()

        client = get_weaviate_client()
        collection = client.collections.get(COLLECTION_NAME)
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        client.close()

        if not results.objects:
            return "No relevant documents found. Run: python scripts/ingest.py first."

        parts = [f"Found {len(results.objects)} relevant chunk(s):\n"]
        for i, obj in enumerate(results.objects, 1):
            props = obj.properties
            dist = obj.metadata.distance if obj.metadata else None
            sim = round(1 - dist, 3) if isinstance(dist, float) else "N/A"
            parts.append(
                f"\n--- Result {i} (similarity: {sim}) ---\n"
                f"Source: {props.get('source', 'Unknown')}\n"
                f"Title: {props.get('title', 'N/A')}\n"
                f"Content:\n{props.get('content', '')}\n"
            )
        return "\n".join(parts)

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"Error searching documents: {e}"


# ── Tool: USGS water quality ─────────────────────────────────────────────────

# Parameter codes for USGS NWIS instantaneous values (/nwis/iv/)
# These are sensor-based, continuously measured parameters — available for most states.
# Lab-only codes (like 00630 nitrate) are NOT in /iv/ — use in-situ codes instead.
_PARAM_CODES = {
    "nitrate":              "99133",  # in-situ nitrate+nitrite sensor, mg/L as N
    "nitrate+nitrite":      "99133",
    "dissolved oxygen":     "00300",
    "ph":                   "00400",
    "turbidity":            "63680",
    "temperature":          "00010",
    "temperature, water":   "00010",
    "conductance":          "00095",
    "specific conductance": "00095",
    "streamflow":           "00060",
    "gage height":          "00065",
}

async def get_water_quality_data(
    state_code: str,
    characteristic: str,
    site_type: str = "Stream",
    count_only: bool = False,
) -> str:
    """Fetch real-time water quality data from USGS NWIS instantaneous values API."""
    state = state_code.upper()

    # Resolve parameter code
    param_code = _PARAM_CODES.get(characteristic.lower())
    if not param_code:
        for key, code in _PARAM_CODES.items():
            if key in characteristic.lower() or characteristic.lower() in key:
                param_code = code
                break
    if not param_code:
        return (
            f"Unknown characteristic '{characteristic}'. "
            f"Supported parameters: {', '.join(_PARAM_CODES.keys())}"
        )

    # Use instantaneous values endpoint — sensor data, always current
    url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format":      "json",
        "stateCd":     state.lower(),
        "parameterCd": param_code,
        "siteType":    "ST",          # ST = stream/river
        "period":      "P30D",        # last 30 days
        "siteStatus":  "active",
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=10.0),
            follow_redirects=True,
        ) as client:
            response = await client.get(url, params=params)

        if response.status_code != 200:
            return (
                f"USGS NWIS returned status {response.status_code}. "
                f"Check https://waterservices.usgs.gov for service status."
            )

        data = response.json()
        time_series = data.get("value", {}).get("timeSeries", [])

        if not time_series:
            return (
                f"No active USGS sensor stations found for '{characteristic}' "
                f"(param code {param_code}) in {state} streams in the last 30 days. "
                f"Note: nitrate sensors are only deployed at select USGS stations. "
                f"Try 'dissolved oxygen', 'ph', 'temperature', or 'streamflow' "
                f"for broader coverage."
            )

        all_values = []
        station_summaries = []

        for ts in time_series[:10]:
            site_name  = ts.get("sourceInfo", {}).get("siteName", "Unknown site")
            site_code  = ts.get("sourceInfo", {}).get("siteCode", [{}])[0].get("value", "")
            unit       = ts.get("variable", {}).get("unit", {}).get("unitCode", "")
            values_raw = ts.get("values", [{}])[0].get("value", [])

            numeric = []
            for v in values_raw:
                try:
                    val = float(v["value"])
                    numeric.append(val)
                except (ValueError, TypeError, KeyError):
                    continue

            if numeric:
                all_values.extend(numeric)
                latest = numeric[-1]
                station_summaries.append(
                    f"  • {site_name} (USGS-{site_code})\n"
                    f"    Latest: {latest:.2f} {unit} | "
                    f"Min: {min(numeric):.2f} | Max: {max(numeric):.2f} | "
                    f"Mean: {sum(numeric)/len(numeric):.2f} | Readings: {len(numeric)}"
                )

        if not all_values:
            return (
                f"Stations found for '{characteristic}' in {state} but no numeric "
                f"sensor readings in the last 30 days. Data may be provisional."
            )

        unit = time_series[0].get("variable", {}).get("unit", {}).get("unitCode", "") if time_series else ""

        lines = [
            f"USGS NWIS Real-Time Data — {characteristic} in {state} (streams, last 30 days)",
            f"Parameter code: {param_code} | Unit: {unit}",
            f"Active sensor stations reporting: {len(station_summaries)}",
            f"Overall — Min: {min(all_values):.2f} | Max: {max(all_values):.2f} | "
            f"Mean: {sum(all_values)/len(all_values):.2f} {unit}",
            "\nPer-station breakdown (up to 10 stations):",
        ]
        lines.extend(station_summaries)
        return "\n".join(lines)

    except httpx.TimeoutException:
        return (
            f"USGS NWIS timed out for '{characteristic}' in {state}. "
            f"Try again or visit https://waterservices.usgs.gov"
        )
    except Exception as e:
        logger.error(f"USGS API error: {e}")
        return f"Error calling USGS NWIS API: {e}"


async def get_epa_facility_info(
    state_code: str,
    program: str = "NPDES",
    limit: int = 10,
) -> str:
    """Look up EPA-registered facilities via the ECHO Facility Registry."""
    state = state_code.upper()
    limit = min(limit, 50)
    url = "https://echodata.epa.gov/echo/facility_search.json"
    params = {
        "output": "JSON",
        "p_st": state,
        "p_act": "Y",
        "qcolumns": "1,3,4,5,6,9,10,13",
        "responseset": limit,
    }
    if program == "NPDES":
        params["p_ptype"] = "NPD"
    elif program == "RCRA":
        params["p_ptype"] = "RCR"
    elif program == "SDWIS":
        params["p_ptype"] = "SDW"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)

        if response.status_code != 200:
            return f"EPA API error {response.status_code}"

        data = response.json()
        facilities = (
            data.get("Results", {}).get("Facilities", [])
            or data.get("Facilities", [])
            or []
        )

        if not facilities:
            return f"No active {program} facilities found in {state}."

        lines = [f"EPA {program} Facilities in {state} (showing {len(facilities)}):"]
        for i, f in enumerate(facilities[:limit], 1):
            lines.append(
                f"{i}. {f.get('FacilityName', 'N/A')}\n"
                f"   Address: {f.get('LocationAddress', '')}, {f.get('CityName', '')}, {state}\n"
                f"   Lat/Lon: {f.get('Latitude83', 'N/A')}, {f.get('Longitude83', 'N/A')}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"EPA API error: {e}")
        return f"Error calling EPA API: {e}"


# ── Helpers ───────────────────────────────────────────────────────────────────

_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09",
    "DE":"10","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18",
    "IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25",
    "MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32",
    "NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39",
    "OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47",
    "TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55",
    "WY":"56","DC":"11",
}

def _state_fips(code: str) -> str:
    return _FIPS.get(code.upper(), "48")


# ── Tool registry (used by agent.py) ─────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_environmental_docs",
            "description": (
                "Search through indexed environmental and water-treatment documents "
                "(EPA reports, WHO guidelines, Ecolab sustainability data, Legionella "
                "control, wastewater treatment) using semantic similarity. Use this to "
                "answer questions about water quality standards, treatment processes, "
                "sustainability goals, or environmental regulations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question or topic to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_water_quality_data",
            "description": (
                "Retrieve real-time water quality monitoring data from the USGS Water "
                "Quality Portal. Returns pH, dissolved oxygen, turbidity, nitrates, "
                "temperature, conductance, etc. for a given US state and water body type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "state_code": {
                        "type": "string",
                        "description": "Two-letter US state code (e.g. 'TX', 'CA', 'FL').",
                    },
                    "characteristic": {
                        "type": "string",
                        "description": (
                            "Water quality parameter. Examples: 'pH', 'Dissolved oxygen', "
                            "'Turbidity', 'Nitrate', 'Temperature, water', 'Conductance'."
                        ),
                    },
                    "site_type": {
                        "type": "string",
                        "description": "Monitoring site type: 'Stream', 'Lake', 'Well', 'Estuary'.",
                    },
                },
                "required": ["state_code", "characteristic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_epa_facility_info",
            "description": (
                "Look up active EPA-regulated facilities (industrial dischargers, water "
                "treatment plants) in a US state using the EPA ECHO Facility Registry. "
                "Supports NPDES (water discharge), RCRA (hazardous waste), SDWIS (drinking water)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "state_code": {
                        "type": "string",
                        "description": "Two-letter US state code.",
                    },
                    "program": {
                        "type": "string",
                        "description": "EPA program: 'NPDES', 'RCRA', or 'SDWIS'.",
                    },
                },
                "required": ["state_code"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_environmental_docs": search_environmental_docs,
    "get_water_quality_data":    get_water_quality_data,
    "get_epa_facility_info":     get_epa_facility_info,
}
