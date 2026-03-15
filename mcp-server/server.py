"""
Ecolab AI Agent - MCP Server
Exposes RAG and API tools for the Claude deepagent CLI agent.
"""

import asyncio
import json
import os
import sys
import logging
from typing import Any

import httpx
import weaviate
import numpy as np
from weaviate.classes.query import MetadataQuery
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("ecolab-mcp")

# ── Weaviate Embedded + local embedder ──────────────────────────────────────
COLLECTION_NAME = "EcolabDocs"
WEAVIATE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".weaviate_data")

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        _embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        logger.info("Embedding model ready.")
    return _embedder

def get_weaviate_client() -> weaviate.WeaviateClient:
    os.makedirs(WEAVIATE_DATA_DIR, exist_ok=True)
    from weaviate.embedded import EmbeddedOptions
    options = EmbeddedOptions(
        persistence_data_path=WEAVIATE_DATA_DIR,
        port=8079,
        grpc_port=50052,
        additional_env_vars={
            "ENABLE_MODULES": "",
            "DEFAULT_VECTORIZER_MODULE": "none",
        },
    )
    client = weaviate.WeaviateClient(embedded_options=options)
    client.connect()
    return client

# ── MCP Server ───────────────────────────────────────────────────────────────
app = Server("ecolab-agent")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_environmental_docs",
            description=(
                "Search through indexed environmental and water-treatment documents "
                "(EPA reports, scientific literature, Ecolab sustainability data) "
                "using semantic similarity. Use this to answer questions about water "
                "quality standards, treatment processes, sustainability goals, or "
                "environmental regulations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question or topic to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_water_quality_data",
            description=(
                "Retrieve real-time water quality monitoring data from the USGS "
                "Water Quality Portal. Returns measurements like pH, dissolved oxygen, "
                "turbidity, nitrates, temperature, etc. for a given US location. "
                "Use this when the user asks about actual water quality readings, "
                "site-specific measurements, or wants to compare standards against "
                "real data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "state_code": {
                        "type": "string",
                        "description": "Two-letter US state code (e.g. 'TX', 'CA', 'FL').",
                    },
                    "characteristic": {
                        "type": "string",
                        "description": (
                            "Water quality characteristic to query. Examples: "
                            "'pH', 'Dissolved oxygen', 'Turbidity', 'Nitrate', "
                            "'Temperature, water', 'Conductance'."
                        ),
                    },
                    "site_type": {
                        "type": "string",
                        "description": "Type of monitoring site: 'Stream', 'Lake', 'Well', 'Estuary'.",
                        "default": "Stream",
                    },
                    "count_only": {
                        "type": "boolean",
                        "description": "If true, return only the count of available records.",
                        "default": False,
                    },
                },
                "required": ["state_code", "characteristic"],
            },
        ),
        Tool(
            name="get_epa_facility_info",
            description=(
                "Look up EPA-registered facilities (industrial plants, water treatment "
                "facilities, etc.) in a given US state using the EPA Facility Registry "
                "Service. Returns facility names, locations, and environmental program "
                "affiliations. Useful for questions about industrial water dischargers "
                "or regulated facilities near a location."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "state_code": {
                        "type": "string",
                        "description": "Two-letter US state code (e.g. 'TX', 'CA').",
                    },
                    "program": {
                        "type": "string",
                        "description": (
                            "EPA program filter. Options: 'NPDES' (water discharge permits), "
                            "'RCRA' (hazardous waste), 'SDWIS' (drinking water)."
                        ),
                        "default": "NPDES",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max facilities to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["state_code"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    if name == "search_environmental_docs":
        return await _rag_search(arguments)
    elif name == "get_water_quality_data":
        return await _usgs_water_quality(arguments)
    elif name == "get_epa_facility_info":
        return await _epa_facility(arguments)
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")]
        )


# ── Tool Implementations ─────────────────────────────────────────────────────

async def _rag_search(args: dict) -> CallToolResult:
    query = args["query"]
    top_k = args.get("top_k", 5)

    try:
        # Embed the query locally using sentence-transformers
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
            return CallToolResult(
                content=[TextContent(type="text", text="No relevant documents found in the knowledge base. Make sure you have run: python scripts/ingest.py")]
            )

        output_parts = [f"Found {len(results.objects)} relevant document chunk(s):\n"]
        for i, obj in enumerate(results.objects, 1):
            props = obj.properties
            distance = obj.metadata.distance if obj.metadata else None
            similarity = round(1 - distance, 3) if isinstance(distance, float) else "N/A"
            output_parts.append(
                f"\n--- Result {i} (similarity: {similarity}) ---\n"
                f"Source: {props.get('source', 'Unknown')}\n"
                f"Title: {props.get('title', 'N/A')}\n"
                f"Content:\n{props.get('content', '')}\n"
            )

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(output_parts))]
        )

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error searching documents: {str(e)}\nMake sure you have run: python scripts/ingest.py")]
        )


async def _usgs_water_quality(args: dict) -> CallToolResult:
    state = args["state_code"].upper()
    characteristic = args["characteristic"]
    site_type = args.get("site_type", "Stream")
    count_only = args.get("count_only", False)

    # USGS Water Quality Portal REST API
    base_url = "https://www.waterqualitydata.us/data/Result/search"
    params = {
        "statecode": f"US:{_state_fips(state)}",
        "characteristicName": characteristic,
        "siteType": site_type,
        "mimeType": "json",
        "dataProfile": "resultPhysChem",
        "providers": "NWIS",
    }

    if count_only:
        params["countOnly"] = "yes"
        params.pop("dataProfile", None)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(base_url, params=params)

        if response.status_code != 200:
            return CallToolResult(
                content=[TextContent(type="text", text=f"USGS API error {response.status_code}: {response.text[:500]}")]
            )

        data = response.json()

        if count_only:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Total records available for '{characteristic}' in {state} ({site_type}): {json.dumps(data, indent=2)}")]
            )

        # Parse and summarize results
        records = data if isinstance(data, list) else data.get("results", [])
        if not records:
            return CallToolResult(
                content=[TextContent(type="text", text=f"No water quality data found for '{characteristic}' in {state}.")]
            )

        # Summarize top 10 readings
        summary_lines = [
            f"💧 USGS Water Quality Data: {characteristic} in {state} ({site_type}s)\n",
            f"Total records retrieved: {len(records)}\n",
        ]

        values = []
        for rec in records[:50]:
            try:
                val = float(rec.get("ResultMeasureValue", "") or "")
                unit = rec.get("ResultMeasure/MeasureUnitCode", "")
                values.append(val)
            except (ValueError, TypeError):
                continue

        if values:
            summary_lines.append(f"Value range: {min(values):.3f} – {max(values):.3f}")
            summary_lines.append(f"Mean value: {sum(values)/len(values):.3f}")
            summary_lines.append(f"Sample count (with numeric values): {len(values)}")

        # Show sample records
        summary_lines.append("\nSample records (up to 5):")
        for rec in records[:5]:
            summary_lines.append(
                f"  • Site: {rec.get('MonitoringLocationIdentifier', 'N/A')}"
                f" | Date: {rec.get('ActivityStartDate', 'N/A')}"
                f" | Value: {rec.get('ResultMeasureValue', 'N/A')} {rec.get('ResultMeasure/MeasureUnitCode', '')}"
                f" | Depth: {rec.get('ActivityDepthHeightMeasure/MeasureValue', 'surface')}"
            )

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(summary_lines))]
        )

    except Exception as e:
        logger.error(f"USGS API error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error calling USGS API: {str(e)}")]
        )


async def _epa_facility(args: dict) -> CallToolResult:
    state = args["state_code"].upper()
    program = args.get("program", "NPDES")
    limit = min(args.get("limit", 10), 50)

    # EPA ECHO Facility Search API
    url = "https://echodata.epa.gov/echo/facility_search.json"
    params = {
        "output": "JSON",
        "p_st": state,
        "p_act": "Y",          # active facilities
        "qcolumns": "1,3,4,5,6,9,10,13",  # facility name, addr, city, state, zip, lat, lon, program
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
            return CallToolResult(
                content=[TextContent(type="text", text=f"EPA API error {response.status_code}")]
            )

        data = response.json()
        facilities = (
            data.get("Results", {}).get("Facilities", [])
            or data.get("Facilities", [])
            or []
        )

        if not facilities:
            return CallToolResult(
                content=[TextContent(type="text", text=f"No active {program} facilities found in {state}.")]
            )

        lines = [f"🏭 EPA {program} Facilities in {state} (showing {len(facilities)}):\n"]
        for i, f in enumerate(facilities[:limit], 1):
            lines.append(
                f"{i}. {f.get('FacilityName', 'N/A')}\n"
                f"   Address: {f.get('LocationAddress', '')}, {f.get('CityName', '')}, {state}\n"
                f"   Lat/Lon: {f.get('Latitude83', 'N/A')}, {f.get('Longitude83', 'N/A')}\n"
            )

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))]
        )

    except Exception as e:
        logger.error(f"EPA API error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error calling EPA API: {str(e)}")]
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    return _FIPS.get(code.upper(), "48")  # default TX


# ── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
