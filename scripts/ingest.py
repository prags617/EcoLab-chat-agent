"""
Ingest environmental documents into Weaviate for RAG.

Uses Weaviate Embedded (no Docker needed) + sentence-transformers for local embeddings.
Vectors are computed in Python and stored directly — no sidecar container required.

Run: python scripts/ingest.py
"""

import os
import sys
import textwrap
import logging
import hashlib

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.embedded import EmbeddedOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

COLLECTION_NAME = "EcolabDocs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
WEAVIATE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".weaviate_data")

DOCUMENTS = [
    {
        "title": "EPA National Primary Drinking Water Regulations",
        "source": "EPA",
        "content": textwrap.dedent("""
            The EPA's National Primary Drinking Water Regulations (NPDWRs) are legally enforceable
            standards that apply to public water systems. These regulations protect public health by
            limiting the levels of contaminants in drinking water.

            Maximum Contaminant Levels (MCLs) — Key Standards:
            - Total Coliform Bacteria: 0 (zero tolerance for public water systems serving >1000/day)
            - Nitrate: 10 mg/L (as N) — high levels cause methemoglobinemia (blue baby syndrome)
            - Nitrite: 1 mg/L (as N)
            - Lead: 0.015 mg/L action level (15 ppb) — no safe level exists; action level triggers treatment
            - Copper: 1.3 mg/L action level
            - Arsenic: 0.010 mg/L (10 ppb) — carcinogen; naturally occurring in some groundwater
            - Fluoride: 4.0 mg/L (secondary standard 2.0 mg/L for cosmetic effects)
            - Chlorine residual (disinfectant): maximum 4.0 mg/L as MRDL
            - Chloramines: maximum 4.0 mg/L as MRDL
            - Trihalomethanes (THMs): 0.080 mg/L (80 ppb) — disinfection byproducts
            - Haloacetic acids (HAA5): 0.060 mg/L (60 ppb)
            - Turbidity: 1 NTU (must not exceed 0.3 NTU in 95% of samples)
            - pH: not regulated at federal level but recommended 6.5-8.5

            Treatment techniques required for:
            - Surface water (filtration + disinfection mandatory)
            - Groundwater under direct influence of surface water
            - Lead and copper (corrosion control treatment programs)

            Secondary standards (non-enforceable, aesthetic):
            - Iron: 0.3 mg/L; Manganese: 0.05 mg/L; Sulfate: 250 mg/L
            - Total Dissolved Solids (TDS): 500 mg/L; Chloride: 250 mg/L; pH: 6.5-8.5

            Water systems must monitor, report, and notify customers of violations. The Safe Drinking
            Water Act (SDWA) authorizes EPA to set these standards and requires systems to use best
            available treatment technology.
        """),
    },
    {
        "title": "Water Treatment Processes: Coagulation, Flocculation and Filtration",
        "source": "EPA Technical Guidance",
        "content": textwrap.dedent("""
            Conventional water treatment involves several sequential processes to remove contaminants
            and produce safe drinking water.

            1. COAGULATION
            Coagulation is the first step in conventional treatment. Chemicals (coagulants) are added
            to water to neutralize the negative charges on particles, colloids, and dissolved organics.
            Common coagulants:
            - Alum (aluminum sulfate): most widely used; effective pH range 5.5-8.0
            - Ferric sulfate / Ferric chloride: effective over wider pH range (4-12); better for
              color removal and cold water
            - Polyaluminum chloride (PAC): pre-hydrolyzed; lower dose needed; less sludge
            - Cationic polymers: used as primary or secondary coagulants

            Jar testing is essential to determine optimal coagulant type and dose.

            2. FLOCCULATION
            After coagulation, gentle mixing promotes collision and aggregation of destabilized particles
            into larger flocs. Flocculation basins typically have 3 stages with decreasing mixing
            intensity (G values 20-80 per second reducing to 5-10 per second). Retention time: 20-40 minutes.

            3. SEDIMENTATION / CLARIFICATION
            Gravity settling removes flocs. Surface overflow rate (SOR): 500-1000 gpd/ft2.
            Dissolved Air Flotation (DAF) is preferred for algae-laden or low-turbidity source waters.

            4. FILTRATION
            Rapid sand filtration removes residual turbidity and pathogens.
            - Dual-media filters (anthracite + sand): most common
            - Target filtered water turbidity: less than 0.1 NTU (best practice); less than 0.3 NTU (regulatory)
            Membrane filtration (MF, UF): provides absolute barrier for Cryptosporidium and Giardia.

            5. DISINFECTION
            Final disinfection inactivates pathogens not removed by filtration.
            - Chlorine: most common; CT values vary by pathogen and pH/temperature
            - Chloramines: secondary disinfection; less THM/HAA formation
            - UV: effective for Cryptosporidium (chlorine resistant); no residual
            - Ozone: powerful oxidant; breaks down NOM; requires post-filtration
            Disinfection byproduct (DBP) management requires balancing microbial safety vs. chemical risk.
        """),
    },
    {
        "title": "Ecolab 2023 Sustainability Goals and Water Stewardship",
        "source": "Ecolab Sustainability Report",
        "content": textwrap.dedent("""
            Ecolab is a global leader in water, hygiene, and infection prevention solutions, serving
            customers in food, healthcare, hospitality, and industrial markets.

            2030 IMPACT GOALS - WATER:
            - Help customers conserve 300 billion gallons of water annually by 2030
            - Achieve water positivity in Ecolab's own operations (return more water than consumed)
            - Deliver water risk solutions to 10,000 high-water-risk locations
            - Support access to safe water for 1 billion people through partnerships

            WATER STEWARDSHIP APPROACH:
            Ecolab's Water Risk Monetizer tool helps businesses quantify the financial risk of water
            scarcity, quality, and flooding at facility level. In 2022, Ecolab helped customers conserve
            218 billion gallons, equivalent to the annual water needs of 700 million people.

            3D TRASAR TECHNOLOGY:
            Ecolab's 3D TRASAR technology provides real-time monitoring and automated treatment of
            cooling water systems. Sensors measure key parameters (pH, conductivity, inhibitor residual,
            corrosion, microbial activity) and automatically adjust chemical dosing. Benefits:
            - Reduces water consumption by up to 20% vs. conventional treatment
            - Prevents scale, corrosion, and biological fouling
            - Minimizes blowdown (water discharge) and chemical waste
            - Remote monitoring via cloud-based dashboard

            FOOD SAFETY AND HYGIENE:
            Ecolab's food safety programs help food processors comply with FSMA regulations. Key products:
            - Peracetic acid (PAA)-based sanitizers: effective against biofilms; low environmental impact
            - Clean-in-place (CIP) systems: automated cleaning of food processing equipment

            INDUSTRIAL WATER TREATMENT:
            Ecolab serves refineries, power plants, chemical plants, and paper mills with:
            - Cooling water treatment (scale, corrosion, microbiological control)
            - Boiler water treatment (oxygen scavenging, scale/corrosion inhibition)
            - Wastewater treatment (coagulants, flocculants, dewatering aids)
            - AI-powered water treatment optimization (predictive dosing)
        """),
    },
    {
        "title": "WHO Guidelines for Drinking Water Quality",
        "source": "World Health Organization",
        "content": textwrap.dedent("""
            The WHO Guidelines for Drinking-water Quality (4th ed., 2022) provide the scientific basis
            for drinking water standards worldwide.

            MICROBIAL QUALITY:
            - E. coli / thermotolerant coliforms: must not be detectable in any 100 mL sample
            - All water intended for drinking must be disinfected
            - Multiple barrier approach: protection of source + treatment + distribution safety
            - Turbidity less than 1 NTU recommended before disinfection for effective UV/chlorine efficacy

            CHEMICAL PARAMETERS (WHO Guideline Values):
            - Nitrate: 50 mg/L (as NO3); note EPA uses 10 mg/L as N which is approximately 44 mg/L as NO3
            - Fluoride: 1.5 mg/L
            - Arsenic: 0.01 mg/L
            - Lead: 0.01 mg/L
            - Chromium: 0.05 mg/L total chromium
            - Manganese: 0.08 mg/L (health-based); 0.1 mg/L (aesthetic)
            - Chlorine residual: 0.2-0.5 mg/L free chlorine (min in distribution)

            WATER SCARCITY CONTEXT:
            - 2 billion people lack safely managed drinking water services (WHO/UNICEF 2023)
            - 3.6 billion people face water scarcity at least one month per year
            - Agriculture accounts for approximately 70% of global freshwater withdrawals
            - Industry and thermoelectric power: approximately 20% of global freshwater

            WATER SAFETY PLANS (WHO recommended approach):
            1. System assessment: identify hazards in catchment, treatment, and distribution
            2. Monitoring: focus on critical control points, not just final water quality
            3. Management: operational procedures, corrective actions, verification monitoring
            4. Supporting programs: training, hygiene promotion, community engagement
        """),
    },
    {
        "title": "Cooling Tower Water Management and Legionella Control",
        "source": "CDC and ASHRAE Guidance",
        "content": textwrap.dedent("""
            Cooling towers are a critical component of HVAC and industrial cooling systems.
            Improper water management creates risk of Legionella pneumophila growth, which causes
            Legionnaires' disease, a severe form of pneumonia.

            LEGIONELLA RISK IN COOLING SYSTEMS:
            - Legionella thrives at 25-45 degrees C (optimal 35-37 degrees C); killed above 60 degrees C
            - Stagnant water, sediment, and biofilm promote growth
            - Aerosol transmission: infected droplets from cooling tower drift
            - CDC estimates 8,000-18,000 hospitalized cases of Legionnaires' disease per year in US

            WATER MANAGEMENT PROGRAM (WMP):
            ASHRAE 188-2018 and CDC guidance require facilities to have documented WMPs:
            1. Building team responsibilities and flow diagram of water systems
            2. Hazard analysis (similar to HACCP for food safety)
            3. Control measures and trigger levels
            4. Corrective actions when limits exceeded
            5. Verification and validation monitoring

            CHEMICAL TREATMENT PARAMETERS:
            - pH: maintain 6.5-9.0; optimal 7.0-8.5 for biocide efficacy
            - Oxidizing biocides: chlorine (0.5-2.0 mg/L free), bromine, chlorine dioxide
            - Non-oxidizing biocides: isothiazolinones, quaternary ammonium compounds, used in rotation
            - Corrosion inhibitors: molybdate, azoles (copper alloys), phosphonates
            - Scale inhibitors: polymeric dispersants, phosphonates
            - Legionella culture testing: quarterly minimum; qPCR for rapid results

            ECOLAB'S ROLE:
            Ecolab is a leading provider of cooling water treatment programs:
            - PATHGUARD biocide programs (oxidizing + non-oxidizing rotation)
            - 3D TRASAR automated monitoring and control
            - Comprehensive risk assessment and WMP development services
            - Real-time remote monitoring for Legionella risk indicators
        """),
    },
    {
        "title": "Industrial Wastewater Treatment Technologies",
        "source": "EPA Technical Reference",
        "content": textwrap.dedent("""
            Industrial wastewater treatment is required before discharge to surface waters (NPDES permit)
            or municipal sewer systems (pretreatment standards).

            REGULATORY FRAMEWORK:
            - Clean Water Act Section 402: NPDES permits regulate industrial discharges
            - Effluent Limitation Guidelines (ELGs): technology-based standards by industry category
            - Key parameters monitored: BOD, COD, TSS, pH, oil and grease, heavy metals, nutrients

            PHYSICAL-CHEMICAL TREATMENT:
            1. Screening and equalization: remove gross solids; buffer flow/load variations
            2. Neutralization: pH adjustment before biological treatment (target 6-9)
            3. Coagulation/flocculation/sedimentation: remove suspended solids and colloids
            4. Oil-water separation: API separators, induced air flotation (IAF), DAF
               - DAF: effective for emulsified oils, suspended solids, algae
            5. Membrane filtration: MF/UF for turbidity; NF/RO for dissolved contaminants/ZLD

            BIOLOGICAL TREATMENT:
            - Activated sludge (aerobic): BOD removal efficiency greater than 90%
            - Sequencing batch reactors (SBR): flexible; good for variable flows
            - Moving bed biofilm reactors (MBBR): compact; resilient to load variation
            - Anaerobic treatment (UASB): energy recovery via biogas; lower sludge production
            - Biological nutrient removal (BNR): nitrogen via nitrification/denitrification

            ADVANCED TREATMENT:
            - Granular activated carbon (GAC): removes recalcitrant organics, taste/odor
            - Advanced oxidation (AOP): UV + H2O2, Fenton's, ozone for micropollutants
            - Ion exchange: selective removal of metals, nitrate, perchlorate
            - Zero Liquid Discharge (ZLD): evaporation/crystallization; eliminates discharge
        """),
    },
    {
        "title": "Water Scarcity and Sustainable Water Management in Industry",
        "source": "World Resources Institute / UN Water",
        "content": textwrap.dedent("""
            Water scarcity is one of the most pressing global challenges. By 2030, global water demand
            is projected to exceed supply by 40%, driven by population growth, climate change, and
            economic development.

            WATER STRESS HOTSPOTS:
            - High water stress regions: Middle East, North Africa, India, Central Asia, US Southwest
            - 17 countries (home to 1/4 of world population) face extremely high water stress (WRI Aqueduct)
            - Agricultural sector: 70% of freshwater withdrawals; irrigation efficiency often less than 50%

            INDUSTRIAL WATER EFFICIENCY STRATEGIES:
            1. Water auditing and metering: baseline measurement; identify losses
            2. Closed-loop cooling systems: recirculating vs. once-through; saves 95%+ of water
            3. Process optimization: reduce rinse volumes, extend chemical baths, optimize CIP
            4. Water reuse/recycling: treat and recycle process water; greywater reuse
            5. Source diversification: rainwater harvesting, treated wastewater, produced water

            CORPORATE WATER STEWARDSHIP:
            - Water footprint accounting: blue (consumed), green (rainwater), grey (polluted)
            - Science-based targets for water (SBTi Water): align withdrawal with basin-level availability
            - CDP Water Security disclosure: investors increasingly require water risk reporting
            - Alliance for Water Stewardship (AWS) certification: third-party verification

            ECOLAB'S WATER RISK MONETIZER:
            A publicly available tool (co-developed with WRI and Microsoft) that quantifies the
            financial risk of water challenges at site level. Inputs include facility location, water
            use volume, and business sector. Outputs are risk in dollar terms for water scarcity,
            flooding, and water quality degradation.

            CIRCULAR WATER ECONOMY:
            - Wastewater-to-resource: nutrients (struvite/MAP precipitation), biogas, reclaimed water
            - Desalination: growing in water-scarce coastal regions; energy-intensive
            - Nature-based solutions: wetlands, green infrastructure for stormwater management
        """),
    },
]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model (multi-qa-MiniLM-L6-cos-v1)...")
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        logger.info("Model loaded.")
        return model
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    return [v.tolist() for v in vectors]


def get_embedded_client() -> weaviate.WeaviateClient:
    os.makedirs(WEAVIATE_DATA_DIR, exist_ok=True)
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


def ingest():
    model = load_embedder()

    logger.info("Starting Weaviate Embedded...")
    client = get_embedded_client()

    if client.collections.exists(COLLECTION_NAME):
        logger.info(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.collections.delete(COLLECTION_NAME)

    logger.info(f"Creating collection '{COLLECTION_NAME}' (BYO vectors)...")
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="title",    data_type=DataType.TEXT),
            Property(name="source",   data_type=DataType.TEXT),
            Property(name="content",  data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.INT),
            Property(name="doc_id",   data_type=DataType.TEXT),
        ],
    )

    collection = client.collections.get(COLLECTION_NAME)
    total_chunks = 0

    for doc in DOCUMENTS:
        doc_id = hashlib.md5(doc["title"].encode()).hexdigest()[:8]
        chunks = chunk_text(doc["content"])
        logger.info(f"Embedding '{doc['title']}' -> {len(chunks)} chunks")

        vectors = embed_texts(model, chunks)

        objects = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            objects.append(wvc.data.DataObject(
                properties={
                    "title":    doc["title"],
                    "source":   doc["source"],
                    "content":  chunk,
                    "chunk_id": i,
                    "doc_id":   doc_id,
                },
                vector=vector,
            ))

        collection.data.insert_many(objects)
        total_chunks += len(chunks)

    logger.info(f"\nIngestion complete: {len(DOCUMENTS)} documents, {total_chunks} total chunks")
    client.close()


if __name__ == "__main__":
    ingest()
