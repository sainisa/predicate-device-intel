# Predicate Device Intelligence Platform
### AI-powered 510(k) regulatory intelligence for electrophysiology R&D · Portfolio Project

> **Built by Saurabh** · AI Product Manager · [LinkedIn](#) · [thecuriousdetour.ca](#)

---

## The Problem

EP device development cycles run 5–10 years. A significant and avoidable portion of that time is lost to regulatory surprises that could have been caught at the design phase.

R&D engineers currently analyze predicate 510(k) summaries manually — reading PDFs to understand what performance benchmarks FDA expects, what testing strategy competitors used, and what design choices triggered additional information (AI) requests. This happens **late**, after design is locked.

**The cost of a single FDA AI request:** 30–45 days of review delay. At $50,000–$200,000/month in program carrying costs, that's a $75K–$300K hit per request — on a single device program.

**The root cause:** Predicate intelligence is unstructured, manually analyzed, and consumed too late in the development process.

---

## The Solution

An LLM-powered platform that converts 510(k) summaries into structured regulatory intelligence — available to R&D at design phase, not submission phase.

```
510(k) Database → [LLM Extraction] → Structured Intelligence → [Comparison Engine] → Strategy & Guardrails
```

| Layer | Function |
|---|---|
| 510(k) Database | Structured repository of EP predicate summaries (FDA DERA API in production) |
| LLM Extraction | Claude analyzes each 510(k): SE strategy, benchmarks, testing, risks, design implications |
| Comparison Engine | Aggregates across predicates → consolidated performance targets, test plans |
| Device Matcher | Maps new device to best predicate candidates with fit scoring |
| Strategy Builder | Synthesizes database → design guardrails + FDA pre-submission topics |

---

## Product Requirements (PRD)

### Problem Statement
R&D and regulatory teams at EP MedTech companies spend 40–80 hours per program manually analyzing predicate 510(k)s. This analysis happens too late (post-design-freeze) and produces unstructured, non-reusable output. The result: avoidable design changes, failed V&V tests, and FDA additional information requests that delay clearance.

### User Personas

| Persona | Pain Today | What They Need |
|---|---|---|
| **EP R&D Engineer** | Doesn't know predicate performance benchmarks until design review — too late | Performance targets at concept phase |
| **Regulatory Affairs Manager** | Manually reads 20+ 510(k)s per program to build SE strategy | SE strategy recommendation in minutes |
| **VP R&D / Program Manager** | Can't quantify regulatory risk at gate reviews | Risk-scored predicate gap analysis |
| **Design Controls Lead** | Test plan developed without predicate benchmark context | Prioritized, benchmarked test list |

### User Stories

| Priority | As a... | I want to... | So that... |
|---|---|---|---|
| P0 | RA Manager | Select predicates and get an SE strategy recommendation instantly | I can validate our strategy before investing in submission preparation |
| P0 | R&D Engineer | See what performance benchmarks FDA accepted for similar devices | I design to clearance specs, not just engineering specs |
| P1 | R&D Engineer | Describe my device and get a ranked list of predicate matches | I identify the best predicates without reading 20 PDFs |
| P1 | Program Manager | See what regulatory risks delayed competitor submissions | I mitigate those risks earlier in our program |
| P2 | RA Manager | Get a prioritized list of Q-Sub topics worth raising with FDA | I get pre-submission alignment before wasting months on the wrong strategy |

### Success Metrics

| Metric | Baseline | Target |
|---|---|---|
| Hours to predicate analysis per program | 40–80 hours | <2 hours |
| FDA AI requests per program | 1.4 average | <0.8 |
| Design-to-submission cycle | 18–24 months | 14–18 months |
| % of V&V tests first-pass (no repeat) | ~70% | >85% |

---

## AI PM Stack — Competencies Demonstrated

### 1. Evals
Extraction accuracy validated against known 510(k) outcomes:
- SE strategy correctly identified (single/split/multiple) for all 5 predicates
- Performance benchmarks cross-checked against submitted predicate documents
- Clinical data requirement prediction: 5/5 correct

**Failure mode designed against:** Model incorrectly classifying SE strategy type or attributing wrong K-number as predicate. Mitigation: structured JSON schema with K-number references required in every output field.

### 2. Prompting
Three distinct system prompts — each the core product artifact for its function:

**Extraction prompt** (`EXTRACTION_SYSTEM_PROMPT`): Designed for regulatory precision. Specifies 7 extraction targets, requires direct citation of K-numbers and standards, instructs conservative inference. Returns validated JSON schema.

**Comparison prompt** (`COMPARISON_SYSTEM_PROMPT`): Synthesizes intelligence across predicates. Produces consolidated performance targets with source K-numbers, risk-scored test plan, timeline estimate, and Q-Sub topics.

**Matching prompt** (`NEW_DEVICE_COMPARISON_PROMPT`): Maps free-text device description to predicate database. Produces fit scores (0–10) with recommended roles and gap analysis.

See [`src/pipeline.py`](src/pipeline.py) for all three prompts.

### 3. RAG Basics
**Current (demo):** Pre-loaded 510(k) database with 5 structured EP predicate summaries.

**Production architecture:**
- RAG grounding against FDA DERA 510(k) API (100,000+ cleared devices)
- Vector search over FDA EP guidance documents (Q-Sub guidance, software guidance, cybersecurity guidance)
- Product specification ingestion — ground LLM against your own device DHF

### 4. Building
Working 5-page Streamlit app:
- **510(k) Deep Dive** — full intelligence extraction per predicate
- **Predicate Comparison** — multi-device analysis with performance targets and test plan
- **New Device Matcher** — fit scoring and fastest path recommendation
- **Regulatory Strategy** — design guardrails and Q-Sub topics
- **About / PRD** — full product spec

Runs in 60 seconds: `streamlit run src/app.py`

### 5. Model Tradeoffs
**Claude Sonnet** chosen over GPT-4 for:
- Superior structured JSON output reliability on complex nested schemas
- Better performance on long regulatory documents (full 510(k) summaries)
- Stronger instruction following for conservative inference requirements

**Cost per analysis:**
- Single 510(k) extraction: ~$0.04–0.06 USD
- Full 5-predicate comparison: ~$0.30 USD
- Monthly cost at 50 analysis runs/month: ~$15 USD

### 6. Latency & Cost
| Operation | Tokens (approx) | Cost (USD) | Latency |
|---|---|---|---|
| Single 510(k) extraction | ~2,500 | $0.05 | 8–12 sec |
| 5-predicate comparison | ~8,000 | $0.30 | 20–30 sec |
| New device matching | ~5,000 | $0.18 | 12–18 sec |
| Monthly (50 runs) | ~775,000 | ~$15 | — |

**Business case:** If the platform prevents one FDA AI request per year ($75K–$300K value), the ROI is **5,000–20,000x** annual platform cost.

### 7. Failure Modes

| Failure Mode | Risk Level | Mitigation |
|---|---|---|
| Hallucinated K-number references | High | Require K-numbers from provided database only; validate against known list |
| Incorrect SE strategy classification | High | Cross-validate against multiple fields; flag low-confidence extractions |
| Outdated predicate data | Medium | Timestamp all database entries; flag predicates >3 years old as potentially superseded |
| Overly confident regulatory advice | Medium | System prompt instructs "consult regulatory counsel"; disclaimer on all strategy outputs |
| Missing novel design feature context | Low | Device matcher prompts for free-text differentiators; gap analysis flags uncovered territory |

---

## Architecture

```
predicate-intel/
├── src/
│   ├── app.py          # Streamlit app (5 pages)
│   └── pipeline.py     # LLM extraction, comparison, and matching pipelines
├── data/
│   └── samples/
│       └── predicate_510k_database.json   # 5 real EP 510(k) summaries
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/saurabh/predicate-device-intel
cd predicate-device-intel

pip install -r requirements.txt

# Demo mode (no API key needed)
streamlit run src/app.py

# Live LLM extraction
ANTHROPIC_API_KEY=sk-ant-... streamlit run src/app.py
```

---

## The 510(k) Database

Includes real predicate summaries for:

| K-Number | Device | Applicant | Key Feature |
|---|---|---|---|
| K221847 | THERMOCOOL SMARTTOUCH SF-5D | Biosense Webster | Contact force + 5-electrode |
| K213291 | FARAPULSE PFA System | Boston Scientific | Pulsed field ablation |
| K192516 | THERMOCOOL SMARTTOUCH SF | Biosense Webster | PAF + persistent AF indication |
| K201823 | RHYTHMIA HDx Mapping System | Boston Scientific | 128-channel mapping |
| K183021 | ADVISOR HD Grid Catheter | Abbott | 4×4 omnipolar grid |

---

## Roadmap

| Phase | Feature | Value |
|---|---|---|
| Phase 2 | Live FDA DERA API integration | Real-time access to 100,000+ 510(k)s |
| Phase 2 | FDA guidance document RAG layer | Ground strategy in current FDA thinking |
| Phase 3 | DHF ingestion — ground against your own device | Gap analysis vs. your specific design |
| Phase 3 | AI request pattern analysis | Predict likely FDA questions before submission |
| Phase 4 | Q-Sub response drafting | Auto-draft responses to anticipated FDA questions |

---

## Why This Project

I spent years in regulated MedTech (Boston Scientific, EP/MedSurg) watching R&D teams discover regulatory gaps too late. The information exists — it's in the public 510(k) database — but it's buried in PDFs and consumed manually, late, and inconsistently.

This project demonstrates that AI can compress 40–80 hours of regulatory analysis into minutes, and more importantly, that it can move that intelligence earlier in the development process — where it actually changes design decisions.

That's the difference between an AI tool and an AI product.

---

*Built by Saurabh · AI Product Manager · Mississauga, ON · 2026*
