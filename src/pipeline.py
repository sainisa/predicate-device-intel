"""
Predicate Device Intelligence — Analysis Pipeline
LLM-powered extraction and comparison of 510(k) predicate strategies.
"""

import json
import re

# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS — core PM artifacts
# ─────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """
You are a regulatory intelligence analyst specializing in FDA 510(k) submissions for
electrophysiology (EP) medical devices.

Your job is to analyze 510(k) summaries and extract structured regulatory intelligence
that helps R&D and regulatory teams design their products and submissions for faster clearance.

Extract the following. Be precise and conservative — only extract what is explicitly stated.

1. SUBSTANTIAL_EQUIVALENCE_STRATEGY: How did the applicant establish SE?
   - predicate_approach: "single" | "split" | "multiple"
   - intended_use_predicate: which K-number was used for intended use
   - tech_char_predicate: which K-number(s) for technological characteristics
   - key_argument: 1-2 sentence summary of the SE argument
   - clinical_data_required: true | false
   - clinical_data_rationale: why clinical data was or wasn't required

2. PERFORMANCE_BENCHMARKS: Quantitative performance thresholds established.
   Each benchmark: { parameter, value, standard, significance }
   significance: "Critical" | "Important" | "Supporting"

3. TESTING_STRATEGY: What tests were done and in what order of importance.
   Each test: { test_name, standard, category, result_summary, required_for_clearance }
   category: "Electrical Safety" | "Mechanical" | "Biocompatibility" | "Sterilization" | "Software" | "Bench Performance" | "Clinical"

4. REGULATORY_RISK_FLAGS: Factors that complicated or extended the review.
   Each flag: { issue, impact, mitigation_used }

5. CLEARANCE_ACCELERATORS: Factors that helped speed up clearance.
   Each accelerator: { factor, description }

6. DESIGN_IMPLICATIONS: What does this clearance tell an R&D team designing a similar device?
   Each implication: { area, insight, action }
   area: "Design" | "Testing" | "Regulatory Strategy" | "Clinical" | "Software"

7. COMPETITIVE_INTELLIGENCE: What does this reveal about the competitive landscape?
   { market_position, differentiation_claims, performance_bar_set, strategic_notes }

Respond ONLY with valid JSON. No preamble, no markdown.
"""

COMPARISON_SYSTEM_PROMPT = """
You are a regulatory strategy advisor for a MedTech company developing electrophysiology devices.

You will be given structured intelligence extracted from multiple 510(k) predicate device summaries.
Your job is to synthesize this into a strategic regulatory roadmap for a new device in development.

Produce:

1. PREDICATE_SELECTION_RECOMMENDATION:
   - recommended_predicates: list of K-numbers with rationale
   - strategy: "single" | "split" | "multiple"
   - intended_use_predicate: best K-number for intended use
   - tech_char_predicate: best K-number(s) for tech characteristics
   - rationale: why this combination

2. PERFORMANCE_TARGETS: Consolidated performance benchmarks the new device must meet or exceed.
   Each: { parameter, minimum_threshold, recommended_target, source_k_number, risk_if_missed }

3. CRITICAL_TEST_LIST: Prioritized testing plan based on predicate analysis.
   Each: { priority, test, standard, rationale, estimated_weeks }

4. REGULATORY_RISKS: Anticipated review complications and how to mitigate them.
   Each: { risk, likelihood, mitigation_strategy }

5. TIMELINE_ESTIMATE: Realistic 510(k) review timeline estimate.
   { optimistic_days, realistic_days, risk_factors_that_extend }

6. Q_SUBMISSION_TOPICS: Pre-submission questions worth raising with FDA.
   Each: { question, why_important, precedent }

7. DESIGN_GUARDRAILS: What the R&D team must design to — based on predicate analysis.
   Each: { constraint, reason, flexibility }

Respond ONLY with valid JSON. No preamble, no markdown.
"""

NEW_DEVICE_COMPARISON_PROMPT = """
You are a regulatory intelligence analyst for an EP medical device company.

A new device is being evaluated against predicate 510(k)s. Your job is to identify:
1. How closely this new device maps to each predicate
2. Which predicates are strongest for 510(k) strategy
3. What gaps exist that could delay clearance

New Device Description:
{device_description}

Predicate Database:
{predicate_summaries}

Produce a JSON response with:
1. PREDICATE_MATCH_SCORES: For each predicate K-number:
   { k_number, device_name, intended_use_match (0-10), tech_char_match (0-10),
     overall_fit (0-10), strengths, gaps, recommended_role }
   recommended_role: "Primary" | "Split-IntendedUse" | "Split-TechChar" | "Supporting" | "Not Recommended"

2. STRATEGY_RECOMMENDATION: Best overall 510(k) approach given this device and these predicates.

3. KEY_GAPS: What the new device needs to address or test to achieve clearance.

4. FASTEST_PATH: The specific predicate combination most likely to yield fastest review.

Respond ONLY with valid JSON. No preamble, no markdown.
"""


# ─────────────────────────────────────────────────────────────────
# DEMO RESPONSES
# ─────────────────────────────────────────────────────────────────

DEMO_EXTRACTION = {
    "K221847": {
        "substantial_equivalence_strategy": {
            "predicate_approach": "split",
            "intended_use_predicate": "K192516",
            "tech_char_predicate": ["K192516", "K173408"],
            "key_argument": "Established SE through split predicate: intended use from K192516 (same PAF ablation indication), electrode spacing precedent from K173408. FDA accepted that the additional 5th diagnostic electrode is an additive feature, not a new safety question.",
            "clinical_data_required": False,
            "clinical_data_rationale": "Technological similarity to cleared predicate was sufficient. Additional electrode is diagnostic only — no new energy delivery mechanism. Bench data demonstrated equivalent electrical safety profile."
        },
        "performance_benchmarks": [
            {"parameter": "Contact force accuracy", "value": "±1g across 0–200g range", "standard": "Internal BSW TF-2210", "significance": "Critical"},
            {"parameter": "RF power output accuracy", "value": "±5% of set value", "standard": "IEC 60601-2-2", "significance": "Critical"},
            {"parameter": "Thermocouple accuracy", "value": "±1°C", "standard": "IEC 60601-2-2", "significance": "Critical"},
            {"parameter": "Irrigation flow accuracy", "value": "<5% deviation, 2–30 mL/min", "standard": "Internal BSW IF-0042", "significance": "Important"},
            {"parameter": "Deflection range", "value": "180° ± 10° bidirectional", "standard": "Internal BSW DT-1187", "significance": "Important"},
            {"parameter": "Torque transmission ratio", "value": ">0.85", "standard": "Internal BSW DT-1187", "significance": "Supporting"},
            {"parameter": "Tip pull strength", "value": ">10 N at tip/shaft junction", "standard": "ASTM F2602", "significance": "Critical"},
            {"parameter": "Electrode impedance", "value": "<200Ω per electrode", "standard": "Internal", "significance": "Important"},
            {"parameter": "Inter-electrode isolation", "value": ">100kΩ", "standard": "Internal", "significance": "Important"},
            {"parameter": "Sterility assurance level", "value": "SAL 10⁻⁶", "standard": "ISO 11135", "significance": "Critical"}
        ],
        "testing_strategy": [
            {"test_name": "Contact force sensing accuracy", "standard": "Internal BSW TF-2210", "category": "Bench Performance", "result_summary": "±1g across full range — meets spec", "required_for_clearance": True},
            {"test_name": "RF power delivery and temperature", "standard": "IEC 60601-2-2", "category": "Electrical Safety", "result_summary": "Power ±5%, temp ±1°C", "required_for_clearance": True},
            {"test_name": "Tensile strength", "standard": "ASTM F2602", "category": "Mechanical", "result_summary": ">10N without failure", "required_for_clearance": True},
            {"test_name": "Biocompatibility", "standard": "ISO 10993-1", "category": "Biocompatibility", "result_summary": "Pass all endpoints", "required_for_clearance": True},
            {"test_name": "Sterilization validation", "standard": "ISO 11135", "category": "Sterilization", "result_summary": "EO validated, SAL 10⁻⁶", "required_for_clearance": True},
            {"test_name": "Deflection and torque", "standard": "Internal BSW DT-1187", "category": "Mechanical", "result_summary": "180° ±10°, torque >0.85", "required_for_clearance": True},
            {"test_name": "Irrigation flow rate", "standard": "Internal BSW IF-0042", "category": "Bench Performance", "result_summary": "<5% deviation", "required_for_clearance": True},
            {"test_name": "CARTO 3 compatibility — regression", "standard": "IEC 62304", "category": "Software", "result_summary": "No new software — regression only", "required_for_clearance": False}
        ],
        "regulatory_risk_flags": [
            {"issue": "Novel 5th electrode with different spacing (4-4-4 vs 2-5-2 mm)", "impact": "Potential AI request on electrogram signal quality with new spacing", "mitigation_used": "Bench data demonstrating equivalent diagnostic signal quality; precedent from K173408 for spacing variation"},
            {"issue": "Split predicate approach requiring two predicates", "impact": "Increased scrutiny on both predicate relationships", "mitigation_used": "Clear delineation of which predicate covers intended use vs. tech characteristics in submission"}
        ],
        "clearance_accelerators": [
            {"factor": "Strong predicate lineage", "description": "Direct lineage to K192516 (same product family) minimized substantial equivalence questions"},
            {"factor": "No new energy modality", "description": "RF energy — well-established FDA review framework; no new safety questions"},
            {"factor": "No software changes", "description": "CARTO 3 compatibility via regression only — no new software development or IEC 62304 package required"},
            {"factor": "No clinical data required", "description": "Bench-only submission substantially reduces review timeline and cost"}
        ],
        "design_implications": [
            {"area": "Design", "insight": "Additional diagnostic electrodes are acceptable as additive features if energy delivery mechanism is unchanged", "action": "Design new diagnostic features as non-energy-delivering where possible to avoid new safety questions"},
            {"area": "Testing", "insight": "Contact force sensing accuracy (±1g) is the critical performance bar for this product class", "action": "Qualify manufacturing process to ±1g or better before 510(k) submission"},
            {"area": "Regulatory Strategy", "insight": "Split predicate is viable and accepted for this device family", "action": "Map intended use and tech characteristics to separate predicates if single predicate doesn't cover both"},
            {"area": "Design", "insight": "Electrode spacing changes (4-4-4 vs 2-5-2) are acceptable with bench data justification", "action": "Budget bench testing time for any electrode geometry changes vs. predicate"},
            {"area": "Software", "insight": "Reusing cleared software with regression testing avoids IEC 62304 full development package", "action": "Prioritize design compatibility with existing cleared software platform"}
        ],
        "competitive_intelligence": {
            "market_position": "Biosense Webster maintains dominant position in irrigated contact-force ablation catheters",
            "differentiation_claims": "5-electrode diagnostic capability (SF-5D) provides higher-density mapping during ablation without separate mapping catheter exchange",
            "performance_bar_set": "Contact force ±1g, RF power ±5%, irrigation flow <5% deviation are the competitive minimum for this device class",
            "strategic_notes": "BSW's ability to extend cleared platform (SMARTTOUCH SF) with additive features without clinical studies is a significant time-to-market advantage. Competitors entering this space need to match these specs or justify deviation."
        }
    },
    "K213291": {
        "substantial_equivalence_strategy": {
            "predicate_approach": "split",
            "intended_use_predicate": "K192516",
            "tech_char_predicate": ["DEN200054", "K201823"],
            "key_argument": "Most complex SE argument in EP — different energy modality (PFA vs RF). Applicant established that PFA achieves same intended effect (cardiac lesion creation) without new safety questions by referencing de novo DEN200054 as technology class predicate and submitting three clinical studies demonstrating safety profile.",
            "clinical_data_required": True,
            "clinical_data_rationale": "FDA required clinical data due to novel energy modality. Three studies (IMPULSE, PEFCAT, MANIFEST-PF, n=473) were submitted voluntarily to demonstrate safety of irreversible electroporation in cardiac tissue — specifically esophageal and phrenic nerve safety."
        },
        "performance_benchmarks": [
            {"parameter": "Pulse energy accuracy", "value": "±3% of target", "standard": "Internal FP-ELEC-001", "significance": "Critical"},
            {"parameter": "Transmural lesion rate", "value": ">95% of sites in ex vivo model", "standard": "Internal FP-BIO-009", "significance": "Critical"},
            {"parameter": "12-month arrhythmia freedom", "value": "78.5% (clinical)", "standard": "MANIFEST-PF", "significance": "Critical"},
            {"parameter": "Phrenic nerve injury rate", "value": "<1% (0.6% achieved)", "standard": "Clinical endpoints", "significance": "Critical"},
            {"parameter": "Esophageal injury rate", "value": "0% (0 events)", "standard": "Clinical endpoints", "significance": "Critical"},
            {"parameter": "Leakage current", "value": "<10µA", "standard": "IEC 60601-1", "significance": "Critical"},
            {"parameter": "Dielectric withstand", "value": "5kV", "standard": "IEC 60601-1", "significance": "Critical"},
            {"parameter": "Basket deployment cycles", "value": ">1000 without structural failure", "standard": "Internal FP-MECH-012", "significance": "Important"}
        ],
        "testing_strategy": [
            {"test_name": "Pulse delivery accuracy", "standard": "Internal FP-ELEC-001", "category": "Electrical Safety", "result_summary": "±3% energy; waveform within spec", "required_for_clearance": True},
            {"test_name": "Ex vivo lesion characterization — porcine hearts", "standard": "Internal FP-BIO-009", "category": "Bench Performance", "result_summary": ">95% transmural, no esophageal injury", "required_for_clearance": True},
            {"test_name": "EMI/EMC — ICD and pacemaker interference", "standard": "IEC 60601-1-2", "category": "Electrical Safety", "result_summary": "No interference at clinical distances", "required_for_clearance": True},
            {"test_name": "Clinical studies (IMPULSE, PEFCAT, MANIFEST-PF)", "standard": "IDE protocols", "category": "Clinical", "result_summary": "78.5% freedom from arrhythmia at 12mo; 0% esophageal injury", "required_for_clearance": True},
            {"test_name": "Basket mechanical integrity", "standard": "Internal FP-MECH-012", "category": "Mechanical", "result_summary": "1000 cycles without failure", "required_for_clearance": True},
            {"test_name": "Biocompatibility incl. hemolysis", "standard": "ISO 10993-1", "category": "Biocompatibility", "result_summary": "Pass including hemolysis and complement", "required_for_clearance": True},
            {"test_name": "Software — cybersecurity", "standard": "FDA Cybersecurity Guidance 2018", "category": "Software", "result_summary": "Pen testing complete, no critical vulnerabilities", "required_for_clearance": True}
        ],
        "regulatory_risk_flags": [
            {"issue": "Novel energy modality — no established FDA review framework for PFA", "impact": "Two additional information (AI) requests extended review by ~8 weeks", "mitigation_used": "Three clinical studies submitted proactively; de novo DEN200054 cited as technology class anchor"},
            {"issue": "Esophageal safety — major concern with thermal ablation competitors", "impact": "FDA scrutinized ex vivo and clinical esophageal data extensively", "mitigation_used": "3D esophageal model bench testing + zero clinical esophageal events across 473 patients"},
            {"issue": "ICD/pacemaker EMI risk from high-voltage PFA pulses", "impact": "Required dedicated EMI testing not required for RF devices", "mitigation_used": "IEC 60601-1-2 testing plus clinical study monitoring for device interactions"},
            {"issue": "Custom pulse generator — no cleared predicate for generator", "impact": "Full IEC 62304 Class C software package required for generator software", "mitigation_used": "Complete SDLC documentation; FDA pre-submission meeting (Q-Sub) to align on software strategy"}
        ],
        "clearance_accelerators": [
            {"factor": "De novo precedent (DEN200054)", "description": "Farapulse's own de novo for PFA technology class became the predicate — company controlled the regulatory precedent"},
            {"factor": "Proactive multi-study clinical package", "description": "Three studies rather than waiting for FDA to request data — built confidence and compressed back-and-forth"},
            {"factor": "Tissue selectivity safety story", "description": "Phrenic nerve (0.6%) and esophageal (0%) injury data was compelling vs. RF historical rates — FDA recognized this as net safety benefit"}
        ],
        "design_implications": [
            {"area": "Regulatory Strategy", "insight": "New energy modalities require clinical data regardless of bench performance — budget for IDE study", "action": "If developing PFA or other novel energy, initiate IDE strategy and clinical sites 18–24 months before planned 510(k)"},
            {"area": "Design", "insight": "Tissue selectivity (sparing esophagus/phrenic nerve) is a regulatory and commercial advantage that must be demonstrated in ex vivo models", "action": "Include 3D anatomical model esophageal testing in V&V plan from design phase"},
            {"area": "Testing", "insight": "PFA pulse generators require IEC 62304 Class C software — higher documentation burden than RF generators", "action": "Classify generator software early; begin SDLC documentation at software architecture phase"},
            {"area": "Regulatory Strategy", "insight": "Pursuing de novo first (then using it as predicate) gives company control over the technology class definition", "action": "For breakthrough energy modalities, evaluate de novo pathway before 510(k)"}
        ],
        "competitive_intelligence": {
            "market_position": "FARAPULSE established PFA as a new device class in EP — first-mover advantage in irreversible electroporation ablation",
            "differentiation_claims": "Non-thermal mechanism, tissue selectivity, single-application per PV, 40-min procedure time reduction vs RF",
            "performance_bar_set": "78.5% 12-month freedom from arrhythmia, 0% esophageal injury, <1% phrenic nerve injury are the PFA competitive benchmarks",
            "strategic_notes": "Any competitor entering PFA must now match or exceed these clinical endpoints. The esophageal safety bar is essentially 0% — a single event in a competitor's clinical study would be catastrophic from an FDA and commercial standpoint."
        }
    }
}

DEMO_COMPARISON = {
    "predicate_selection_recommendation": {
        "recommended_predicates": ["K192516", "K221847", "K213291"],
        "strategy": "split",
        "intended_use_predicate": "K192516",
        "tech_char_predicate": ["K221847", "K213291"],
        "rationale": "K192516 (THERMOCOOL SMARTTOUCH SF) provides the strongest intended use predicate for AF ablation — broadest indication (PAF + persistent AF), well-established in FDA review. K221847 provides technical precedent for contact force sensing accuracy benchmarks. K213291 is relevant only if the new device uses PFA energy — if RF, exclude from tech char predicate to avoid unnecessary scrutiny of energy modality differences."
    },
    "performance_targets": [
        {"parameter": "Contact force accuracy (if CF-sensing device)", "minimum_threshold": "±2g", "recommended_target": "±1g", "source_k_number": "K221847", "risk_if_missed": "High — FDA will compare directly to BSW predicate; worse accuracy = new safety question"},
        {"parameter": "RF power output accuracy", "minimum_threshold": "±10%", "recommended_target": "±5%", "source_k_number": "K221847", "risk_if_missed": "Medium — affects lesion reproducibility claims"},
        {"parameter": "Tip pull strength", "minimum_threshold": ">8N", "recommended_target": ">10N", "source_k_number": "K221847", "risk_if_missed": "High — structural integrity is critical safety parameter"},
        {"parameter": "Electrode impedance", "minimum_threshold": "<500Ω", "recommended_target": "<200Ω", "source_k_number": "K221847", "risk_if_missed": "Medium — affects signal quality claims"},
        {"parameter": "Sterility assurance level", "minimum_threshold": "SAL 10⁻⁶", "recommended_target": "SAL 10⁻⁶", "source_k_number": "All", "risk_if_missed": "Critical — non-negotiable regulatory requirement"},
        {"parameter": "Biocompatibility (ISO 10993-1)", "minimum_threshold": "Pass all applicable endpoints", "recommended_target": "Pass all endpoints incl. hemolysis", "source_k_number": "All", "risk_if_missed": "Critical — biocompatibility failure blocks clearance"}
    ],
    "critical_test_list": [
        {"priority": 1, "test": "Biocompatibility (ISO 10993-1)", "standard": "ISO 10993-1", "rationale": "Non-negotiable; longest lead time (8–12 weeks for in vivo studies). Start first.", "estimated_weeks": 12},
        {"priority": 2, "test": "Sterilization validation (EO)", "standard": "ISO 11135", "rationale": "Required; 10–14 week validation timeline. Must be on cleared package.", "estimated_weeks": 12},
        {"priority": 3, "test": "Electrical safety — RF generator", "standard": "IEC 60601-2-2", "rationale": "Critical safety standard; defines power output accuracy requirements.", "estimated_weeks": 4},
        {"priority": 4, "test": "Contact force accuracy (if applicable)", "standard": "Internal / BSW benchmark", "rationale": "Key performance differentiator; must meet or exceed ±1g predicate benchmark.", "estimated_weeks": 3},
        {"priority": 5, "test": "Tensile strength — tip/shaft junction", "standard": "ASTM F2602", "rationale": "Structural integrity critical safety parameter; >10N target.", "estimated_weeks": 2},
        {"priority": 6, "test": "Deflection and torque", "standard": "Internal", "rationale": "Handling performance claim; 180° ±10° is established predicate benchmark.", "estimated_weeks": 2},
        {"priority": 7, "test": "Irrigation flow accuracy (if irrigated)", "standard": "Internal", "rationale": "Steam pop prevention; <5% deviation required.", "estimated_weeks": 2},
        {"priority": 8, "test": "Electrode impedance", "standard": "Internal", "rationale": "Signal quality; <200Ω predicate benchmark.", "estimated_weeks": 1},
        {"priority": 9, "test": "Mapping system compatibility", "standard": "IEC 62304 regression", "rationale": "Required if using existing cleared software platform.", "estimated_weeks": 3},
        {"priority": 10, "test": "EMI/EMC", "standard": "IEC 60601-1-2", "rationale": "Required for active device; ICD/pacemaker interference testing.", "estimated_weeks": 3}
    ],
    "regulatory_risks": [
        {"risk": "Novel electrode geometry vs. predicate", "likelihood": "Medium", "mitigation_strategy": "Document electrode spacing rationale with bench data; cite K173408 and K221847 precedents for geometry variation"},
        {"risk": "Software changes to mapping system compatibility", "likelihood": "Low-Medium", "mitigation_strategy": "Limit software changes to regression-testable updates; avoid new IEC 62304 Class C package if possible"},
        {"risk": "Clinical data request from FDA for indication expansion", "likelihood": "High if expanding beyond PAF indication", "mitigation_strategy": "Match predicate indication exactly for first submission; expand indication in subsequent 510(k) with PRECEPT-style data"},
        {"risk": "Contact force sensing accuracy below predicate benchmark", "likelihood": "Low if designed to spec", "mitigation_strategy": "Design to ±1g target with ±1.5g acceptance limit; build margin into manufacturing spec"}
    ],
    "timeline_estimate": {
        "optimistic_days": 90,
        "realistic_days": 130,
        "risk_factors_that_extend": [
            "FDA AI request on novel design features (+30–45 days typical)",
            "Clinical data request if indication exceeds predicate (+6–18 months)",
            "Software IEC 62304 Class C review (+3–6 weeks)",
            "Cybersecurity review if connected device (+2–4 weeks)",
            "Biocompatibility concerns requiring additional studies (+8–12 weeks)"
        ]
    },
    "q_submission_topics": [
        {"question": "Is the proposed split-predicate approach (K192516 for intended use, K221847 for contact force tech char) acceptable?", "why_important": "Confirming predicate strategy before submission avoids major rework if FDA disagrees", "precedent": "BSW successfully used same split in K221847"},
        {"question": "Is bench-only testing sufficient for the modified electrode geometry, or will clinical data be required?", "why_important": "Clinical data requirement adds 18–24 months to timeline; must be resolved early", "precedent": "K221847 cleared on bench data for electrode addition; K192516 required clinical for indication expansion"},
        {"question": "What cybersecurity documentation is required if the device communicates with the mapping system via Bluetooth?", "why_important": "Cybersecurity scope is evolving rapidly; pre-alignment prevents late AI requests", "precedent": "K201823 cybersecurity review extended timeline by ~3 weeks"}
    ],
    "design_guardrails": [
        {"constraint": "Contact force sensing accuracy must be ±1g or better", "reason": "BSW predicate has established this as the performance floor; anything worse = new safety question", "flexibility": "Low — this is a competitive and regulatory minimum"},
        {"constraint": "Intended use must match predicate exactly for bench-only path", "reason": "Any indication expansion triggers clinical data requirement (see PRECEPT precedent)", "flexibility": "None for first 510(k); expand indication in v2 with clinical data"},
        {"constraint": "Use existing cleared mapping system software where possible", "reason": "Avoiding IEC 62304 Class C documentation saves 3–6 months of development and review time", "flexibility": "Medium — minor software updates with regression testing are acceptable"},
        {"constraint": "Biocompatibility testing (ISO 10993-1) must be initiated at design freeze", "reason": "12-week lead time is on the critical path to submission; late start delays submission", "flexibility": "None — this is a scheduling constraint, not a design constraint"},
        {"constraint": "Do not introduce new energy modality without de novo strategy and clinical IDE", "reason": "FARAPULSE required 3 clinical studies and de novo; budget 3–5 years for new energy modality", "flexibility": "None — this is a hard regulatory pathway difference"}
    ]
}

DEMO_MATCH = {
    "predicate_match_scores": [
        {"k_number": "K192516", "device_name": "THERMOCOOL SMARTTOUCH SF", "intended_use_match": 9, "tech_char_match": 7, "overall_fit": 8.5, "strengths": ["Broadest AF indication (PAF + persistent)", "Established contact force sensing precedent", "RF ablation — same energy modality", "Well-accepted by FDA"], "gaps": ["Surround flow irrigation may differ from new device irrigation pattern", "Electrode geometry differences need bench justification"], "recommended_role": "Primary"},
        {"k_number": "K221847", "device_name": "THERMOCOOL SMARTTOUCH SF-5D", "intended_use_match": 8, "tech_char_match": 9, "overall_fit": 8.5, "strengths": ["Most recent clearance — reflects current FDA expectations", "5-electrode design establishes precedent for additional diagnostic electrodes", "Contact force accuracy benchmark (±1g) most current"], "gaps": ["PAF-only indication — narrower than K192516", "Split predicate approach may be questioned if used as sole predicate"], "recommended_role": "Split-TechChar"},
        {"k_number": "K213291", "device_name": "FARAPULSE PFA System", "intended_use_match": 6, "tech_char_match": 3, "overall_fit": 4, "strengths": ["Establishes PFA technology precedent if new device uses PFA energy", "Persistent AF indication useful"], "gaps": ["Different energy modality (PFA vs RF) — if new device is RF, creates confusion", "Required 3 clinical studies — citing this predicate implies FDA may expect similar data", "Custom pulse generator architecture unlikely to match new device"], "recommended_role": "Not Recommended"},
        {"k_number": "K201823", "device_name": "RHYTHMIA HDx Mapping System", "intended_use_match": 4, "tech_char_match": 5, "overall_fit": 4, "strengths": ["Mapping system precedent if new device includes software", "128-channel performance benchmarks established"], "gaps": ["Different device type (mapping system vs ablation catheter)", "Diagnostic-only indication doesn't support ablation intended use"], "recommended_role": "Supporting"},
        {"k_number": "K183021", "device_name": "ADVISOR HD Grid Catheter", "intended_use_match": 4, "tech_char_match": 6, "overall_fit": 5, "strengths": ["Multi-electrode array precedent", "Diagnostic mapping precedent for electrode geometry variation", "Fast review (141 days)"], "gaps": ["Diagnostic-only — no ablation indication", "Grid geometry likely different from new device catheter architecture"], "recommended_role": "Supporting"}
    ],
    "strategy_recommendation": "Use split predicate: K192516 for intended use (broadest AF ablation indication) + K221847 for technological characteristics (most current contact force and electrode specifications). This mirrors the successful strategy used in K221847 itself and presents the strongest case to FDA.",
    "key_gaps": [
        "New electrode geometry vs. predicate must be justified with bench testing data showing equivalent signal quality",
        "If irrigation pattern differs from surround flow, dedicated flow characterization testing required",
        "Any software changes beyond regression testing will require IEC 62304 documentation package",
        "Biocompatibility must cover all new materials introduced vs. predicate"
    ],
    "fastest_path": "K192516 (intended use) + K221847 (tech char) split predicate, bench-only testing, indication limited to PAF. Estimated 90–110 day FDA review if no AI requests. Avoid any indication expansion or software changes that could trigger additional data requests."
}


def extract_510k_intelligence(k_number: str, api_key: str = None, use_demo: bool = True) -> dict:
    if use_demo or not api_key:
        return DEMO_EXTRACTION.get(k_number, DEMO_EXTRACTION["K221847"])

    import anthropic
    # Load the 510k data
    import os, json
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "predicate_510k_database.json")
    with open(data_path) as f:
        db = {d["k_number"]: d for d in json.load(f)}

    device = db.get(k_number)
    if not device:
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Analyze this 510(k) summary:\n\n{json.dumps(device, indent=2)}"}]
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def compare_predicates(k_numbers: list, api_key: str = None, use_demo: bool = True) -> dict:
    DEVICE_META = {
        "K221847": {"name": "THERMOCOOL SMARTTOUCH SF-5D", "type": "ablation", "energy": "RF", "indication": "PAF", "has_cf": True, "has_software": False, "review_days": 166},
        "K213291": {"name": "FARAPULSE PFA System", "type": "ablation_system", "energy": "PFA", "indication": "Persistent AF", "has_cf": False, "has_software": True, "review_days": 162},
        "K192516": {"name": "THERMOCOOL SMARTTOUCH SF", "type": "ablation", "energy": "RF", "indication": "PAF + Persistent AF", "has_cf": True, "has_software": False, "review_days": 145},
        "K201823": {"name": "RHYTHMIA HDx Mapping System", "type": "mapping_system", "energy": "Diagnostic", "indication": "Arrhythmia Mapping", "has_cf": False, "has_software": True, "review_days": 171},
        "K183021": {"name": "ADVISOR HD Grid Catheter", "type": "diagnostic", "energy": "Diagnostic", "indication": "EP Mapping", "has_cf": False, "has_software": False, "review_days": 141},
    }
    ALL_TARGETS = [
        {"parameter": "Contact force accuracy", "minimum_threshold": "+-2g", "recommended_target": "+-1g", "source_k_number": "K221847", "risk_if_missed": "High — FDA compares directly to BSW predicate"},
        {"parameter": "RF power output accuracy", "minimum_threshold": "+-10%", "recommended_target": "+-5%", "source_k_number": "K221847", "risk_if_missed": "Medium — affects lesion reproducibility"},
        {"parameter": "Tip pull strength", "minimum_threshold": ">8N", "recommended_target": ">10N", "source_k_number": "K221847", "risk_if_missed": "High — structural integrity critical"},
        {"parameter": "PFA pulse energy accuracy", "minimum_threshold": "+-5%", "recommended_target": "+-3%", "source_k_number": "K213291", "risk_if_missed": "High — critical PFA safety parameter"},
        {"parameter": "Transmural lesion rate ex vivo", "minimum_threshold": ">90%", "recommended_target": ">95%", "source_k_number": "K213291", "risk_if_missed": "High — FDA scrutinizes lesion quality for PFA"},
        {"parameter": "Mapping position accuracy", "minimum_threshold": "<1mm RMS", "recommended_target": "<0.5mm RMS", "source_k_number": "K201823", "risk_if_missed": "High — primary performance claim for mapping system"},
        {"parameter": "Signal SNR mapping channels", "minimum_threshold": ">30dB", "recommended_target": ">40dB", "source_k_number": "K201823", "risk_if_missed": "Medium — affects annotation accuracy"},
        {"parameter": "Omnipolar signal accuracy", "minimum_threshold": "Within 15% of unipolar", "recommended_target": "Within 10%", "source_k_number": "K183021", "risk_if_missed": "Medium — key differentiator for omnipolar claim"},
        {"parameter": "Grid mechanical integrity", "minimum_threshold": ">500 cycles", "recommended_target": ">1000 cycles", "source_k_number": "K183021", "risk_if_missed": "Medium — structural durability for flexible grid"},
        {"parameter": "Sterility assurance level", "minimum_threshold": "SAL 10e-6", "recommended_target": "SAL 10e-6", "source_k_number": "All", "risk_if_missed": "Critical — non-negotiable regulatory requirement"},
        {"parameter": "Biocompatibility ISO 10993-1", "minimum_threshold": "Pass all endpoints", "recommended_target": "Pass all endpoints incl. hemolysis", "source_k_number": "All", "risk_if_missed": "Critical — failure blocks clearance"},
    ]
    ALL_TESTS = [
        {"priority": 1, "test": "Biocompatibility (ISO 10993-1)", "standard": "ISO 10993-1", "rationale": "Non-negotiable; 12-week lead time. Start first.", "estimated_weeks": 12, "sources": ["K221847","K213291","K192516","K201823","K183021"]},
        {"priority": 2, "test": "Sterilization validation (EO)", "standard": "ISO 11135", "rationale": "Required; 10-14 week validation.", "estimated_weeks": 12, "sources": ["K221847","K213291","K192516","K183021"]},
        {"priority": 3, "test": "Electrical safety RF generator", "standard": "IEC 60601-2-2", "rationale": "Critical for RF energy delivery.", "estimated_weeks": 4, "sources": ["K221847","K192516"]},
        {"priority": 3, "test": "PFA pulse delivery accuracy", "standard": "IEC 60601-1", "rationale": "Critical safety for pulsed field energy.", "estimated_weeks": 5, "sources": ["K213291"]},
        {"priority": 4, "test": "Contact force accuracy", "standard": "Internal BSW benchmark", "rationale": "Must meet or exceed +/-1g predicate benchmark.", "estimated_weeks": 3, "sources": ["K221847","K192516"]},
        {"priority": 5, "test": "Tensile strength tip/shaft junction", "standard": "ASTM F2602", "rationale": "Structural integrity >10N target.", "estimated_weeks": 2, "sources": ["K221847","K192516"]},
        {"priority": 5, "test": "Mapping position accuracy", "standard": "Internal cardiac phantom", "rationale": "<0.5mm RMS based on K201823 benchmark.", "estimated_weeks": 4, "sources": ["K201823"]},
        {"priority": 6, "test": "Grid mechanical integrity 1000 cycles", "standard": "Internal", "rationale": "Structural durability for flexible array.", "estimated_weeks": 3, "sources": ["K183021"]},
        {"priority": 6, "test": "EMI/EMC ICD pacemaker interference", "standard": "IEC 60601-1-2", "rationale": "Required for active devices.", "estimated_weeks": 3, "sources": ["K213291","K221847"]},
        {"priority": 7, "test": "Software IEC 62304 and Cybersecurity", "standard": "IEC 62304 / FDA Cybersecurity", "rationale": "Required for software-driven systems.", "estimated_weeks": 8, "sources": ["K213291","K201823"]},
        {"priority": 8, "test": "Ex vivo lesion characterization", "standard": "Internal porcine model", "rationale": "Transmural lesion and esophageal safety for PFA.", "estimated_weeks": 4, "sources": ["K213291"]},
    ]

    selected_set = set(k_numbers)
    filtered_targets = [t for t in ALL_TARGETS if t["source_k_number"] in selected_set or t["source_k_number"] == "All"]
    filtered_tests = [t for t in ALL_TESTS if any(s in selected_set for s in t["sources"])]

    selected_meta = {k: DEVICE_META[k] for k in k_numbers if k in DEVICE_META}
    has_pfa = any(m["energy"] == "PFA" for m in selected_meta.values())
    has_rf = any(m["energy"] == "RF" for m in selected_meta.values())
    has_software = any(m["has_software"] for m in selected_meta.values())
    has_cf_device = any(m["has_cf"] for m in selected_meta.values())

    iu_priority = ["K192516", "K221847", "K213291", "K201823", "K183021"]
    iu_pred = next((k for k in iu_priority if k in selected_set), k_numbers[0])
    tc_preds = [k for k in k_numbers if k != iu_pred][:2]

    fastest = min((DEVICE_META[k]["review_days"] for k in k_numbers if k in DEVICE_META), default=141)
    slowest = max((DEVICE_META[k]["review_days"] for k in k_numbers if k in DEVICE_META), default=171)

    risk_factors = ["FDA AI request on novel design features (+30-45 days)"]
    if has_pfa:
        risk_factors.append("Clinical data required for PFA energy (+6-18 months)")
    if has_software:
        risk_factors += ["IEC 62304 Class C software review (+3-6 weeks)", "Cybersecurity review (+2-4 weeks)"]

    risks = []
    if has_pfa and has_rf:
        risks.append({"risk": "Energy modality mismatch between selected predicates", "likelihood": "High", "mitigation_strategy": "Use RF predicate for intended use only; PFA predicate for technology class reference only"})
    if has_software:
        risks.append({"risk": "IEC 62304 Class C software documentation required", "likelihood": "High", "mitigation_strategy": "Begin SDLC documentation at architecture phase"})
    if has_cf_device:
        risks.append({"risk": "Contact force accuracy below +-1g benchmark", "likelihood": "Medium", "mitigation_strategy": "Design to +-1g target with manufacturing margin"})
    risks.append({"risk": "FDA AI request on design differences vs. predicate", "likelihood": "Medium", "mitigation_strategy": "Pre-submit Q-Sub to align on key differences before 510(k)"})

    qsubs = [{"question": "Is the proposed predicate strategy (" + iu_pred + " for intended use) acceptable?", "why_important": "Confirming predicate before submission avoids major rework", "precedent": "Standard Q-Sub topic for split predicate strategies"}]
    if has_pfa:
        qsubs.append({"question": "Is clinical data required for PFA energy given selected predicates?", "why_important": "Clinical data adds 18-24 months", "precedent": "K213291 required 3 clinical studies"})
    if has_software:
        qsubs.append({"question": "What IEC 62304 software classification applies?", "why_important": "Class C requires full SDLC; misclassification causes AI requests", "precedent": "K201823 and K213291 required Class C"})
    if not has_pfa and not has_software:
        qsubs.append({"question": "Is bench-only testing sufficient for design differences from selected predicates?", "why_important": "Clinical data adds 18-24 months", "precedent": "K183021 and K221847 cleared bench-only"})

    guardrails = [
        {"constraint": "Biocompatibility testing must start at design freeze", "reason": "12-week lead time on critical path", "flexibility": "None"},
        {"constraint": "Intended use must match " + iu_pred + " exactly", "reason": "Indication expansion triggers clinical IDE", "flexibility": "None for first 510(k)"},
    ]
    if has_cf_device:
        guardrails.append({"constraint": "Contact force accuracy must be +-1g or better", "reason": "Predicate sets this as performance floor", "flexibility": "Low"})
    if has_software:
        guardrails.append({"constraint": "Begin IEC 62304 SDLC at architecture phase", "reason": "Class C package takes 6-9 months", "flexibility": "None"})
    if has_pfa:
        guardrails.append({"constraint": "Clinical IDE required before PFA 510(k)", "reason": "K213291 established FDA expectation for clinical data", "flexibility": "None"})

    strategy_text = ("Based on selected predicates (" + ", ".join(k_numbers) + "): use " +
        iu_pred + " for intended use" +
        (" and " + ", ".join(tc_preds) + " for tech char" if tc_preds else "") + ". " +
        ("Clinical data required for PFA energy." if has_pfa else
         "Bench-only path available if indication matches exactly." if has_rf else
         "Software documentation (IEC 62304) is the critical path item."))

    return {
        "predicate_selection_recommendation": {
            "recommended_predicates": k_numbers,
            "strategy": "split" if len(k_numbers) > 1 else "single",
            "intended_use_predicate": iu_pred,
            "tech_char_predicate": tc_preds,
            "rationale": strategy_text
        },
        "performance_targets": filtered_targets,
        "critical_test_list": filtered_tests,
        "regulatory_risks": risks,
        "timeline_estimate": {"optimistic_days": fastest - 10, "realistic_days": slowest + 30, "risk_factors_that_extend": risk_factors},
        "q_submission_topics": qsubs,
        "design_guardrails": guardrails
    }

def match_new_device(device_description: str, api_key: str = None, use_demo: bool = True) -> dict:
    if use_demo or not api_key:
        return DEMO_MATCH
    return DEMO_MATCH


def match_new_device_dynamic(device_description: str, api_key: str = None, use_demo: bool = True) -> dict:
    desc = device_description.lower()
    is_pfa = "pulsed field" in desc or "pfa" in desc
    is_diagnostic = "diagnostic only" in desc or ("mapping catheter" in desc and "ablation" not in desc)
    is_mapping_system = "mapping system" in desc
    is_rf = not is_pfa and not is_diagnostic and not is_mapping_system
    has_persistent = "persistent" in desc
    has_paf = "paroxysmal" in desc or "paf" in desc or (not has_persistent and not is_diagnostic)
    has_cf = "contact force" in desc
    has_multi = "multi-electrode" in desc or "grid" in desc or "array" in desc
    has_omnipolar = "omnipolar" in desc
    has_mapping = "integrated mapping" in desc or is_mapping_system

    scores = []

    iu_192 = 9 if is_rf and (has_paf or has_persistent) else (5 if is_rf else 3)
    tc_192 = 8 if has_cf else (6 if is_rf else 3)
    scores.append({"k_number": "K192516", "device_name": "THERMOCOOL SMARTTOUCH SF",
        "intended_use_match": iu_192, "tech_char_match": tc_192,
        "overall_fit": round(iu_192*0.5 + tc_192*0.5, 1),
        "strengths": (["Broadest AF indication", "RF ablation — same energy"] if is_rf else ["Persistent AF precedent"])
                    + (["Contact force precedent"] if has_cf else []),
        "gaps": (["Energy modality mismatch — PFA vs RF"] if is_pfa else [])
              + (["No CF sensing to reference"] if not has_cf else []),
        "recommended_role": "Primary" if is_rf and iu_192 >= 7 else ("Split-IntendedUse" if iu_192 >= 5 else "Not Recommended")})

    iu_221 = 8 if is_rf and has_paf else (5 if is_rf else 2)
    tc_221 = 9 if has_cf and is_rf else (7 if has_multi else 4)
    scores.append({"k_number": "K221847", "device_name": "THERMOCOOL SMARTTOUCH SF-5D",
        "intended_use_match": iu_221, "tech_char_match": tc_221,
        "overall_fit": round(iu_221*0.4 + tc_221*0.6, 1),
        "strengths": ["Most recent clearance"] + (["CF ±1g benchmark"] if has_cf else []) + (["Multi-electrode precedent"] if has_multi else []),
        "gaps": (["PAF-only — narrower indication"] if has_persistent else []) + (["Energy mismatch"] if is_pfa else []),
        "recommended_role": "Split-TechChar" if tc_221 >= 7 else ("Supporting" if tc_221 >= 5 else "Not Recommended")})

    iu_213 = 8 if is_pfa else (5 if has_persistent else 3)
    tc_213 = 9 if is_pfa else 2
    scores.append({"k_number": "K213291", "device_name": "FARAPULSE PFA System",
        "intended_use_match": iu_213, "tech_char_match": tc_213,
        "overall_fit": round(iu_213*0.5 + tc_213*0.5, 1),
        "strengths": (["PFA technology class predicate", "Tissue selectivity precedent"] if is_pfa else ["Persistent AF indication"]),
        "gaps": (["3 clinical studies required"] if is_pfa else ["Wrong energy modality for RF device", "Not recommended for RF"]),
        "recommended_role": "Primary" if is_pfa else ("Not Recommended" if is_rf else "Supporting")})

    iu_201 = 8 if is_mapping_system else (4 if has_mapping else 2)
    tc_201 = 8 if is_mapping_system else (5 if has_mapping else 3)
    scores.append({"k_number": "K201823", "device_name": "RHYTHMIA HDx Mapping System",
        "intended_use_match": iu_201, "tech_char_match": tc_201,
        "overall_fit": round(iu_201*0.5 + tc_201*0.5, 1),
        "strengths": ["128-channel mapping precedent", "Software Class C package precedent"] if is_mapping_system else ["Software integration precedent"],
        "gaps": [] if is_mapping_system else ["Different device type", "Diagnostic-only — no ablation support"],
        "recommended_role": "Primary" if is_mapping_system else ("Supporting" if has_mapping else "Not Recommended")})

    iu_183 = 8 if is_diagnostic else (4 if has_multi else 2)
    tc_183 = 9 if has_multi or has_omnipolar else (5 if is_diagnostic else 3)
    scores.append({"k_number": "K183021", "device_name": "ADVISOR HD Grid Catheter",
        "intended_use_match": iu_183, "tech_char_match": tc_183,
        "overall_fit": round(iu_183*0.4 + tc_183*0.6, 1),
        "strengths": (["Grid array precedent", "Omnipolar precedent", "141-day fast review"] if is_diagnostic else ["Multi-electrode geometry precedent"] if has_multi else ["Electrode variation precedent"]),
        "gaps": [] if is_diagnostic else ["Diagnostic-only — no ablation indication"],
        "recommended_role": "Primary" if is_diagnostic else ("Split-TechChar" if has_multi or has_omnipolar else "Supporting")})

    scores.sort(key=lambda x: x["overall_fit"], reverse=True)

    if is_pfa:
        strategy = "Use K213291 (FARAPULSE) as primary predicate. Budget for clinical IDE — FDA requires clinical data for PFA energy."
        fastest = "K213291 + K192516 split. Expect 3 clinical studies and 180–240 day review. Start IDE planning now."
        gaps = ["PFA requires clinical IDE — begin planning immediately", "Pulse generator needs IEC 62304 Class C software package", "Ex vivo esophageal safety testing required", "EMI testing with ICD/pacemakers required"]
    elif is_mapping_system:
        strategy = "Use K201823 (RHYTHMIA HDx) as primary. IEC 62304 Class C software is the critical path — start SDLC documentation at architecture phase."
        fastest = "K201823 sole predicate. Bench-only if accuracy meets spec. 90–120 day review if software package complete."
        gaps = ["IEC 62304 Class C — begin SDLC at architecture phase", "Cybersecurity penetration testing before submission", "Mapping accuracy methodology — pre-agree with FDA via Q-Sub"]
    elif is_diagnostic:
        strategy = "Use K183021 (ADVISOR HD Grid) as primary. Bench-only path — fastest device class to clear in EP."
        fastest = "K183021 as sole predicate. Bench-only. 90–141 day review. No clinical IDE needed."
        gaps = ["Electrode geometry differences need bench justification", "Biocompatibility ISO 10993-1 — start at design freeze (12 weeks)"]
    else:
        strategy = ("Split predicate: K192516 (intended use) + K221847 (CF tech char). Bench-only if indication matches exactly." if has_cf
                   else "K192516 as primary predicate for RF ablation. Bench-only path available if indication matches predicate.")
        fastest = ("K192516 + K221847 split, bench-only, indication limited to " + ("PAF + persistent AF" if has_persistent else "PAF") + ". Target 90–110 days." if has_cf
                  else "K192516 sole predicate, bench-only, match indication exactly. 90–120 day target.")
        gaps = ["Biocompatibility ISO 10993-1 — initiate at design freeze (12-week lead time)",
                "Indication expansion beyond predicate triggers clinical IDE requirement",
                "Any electrode geometry change needs bench data vs. predicate"] + (["CF accuracy must meet ±1g benchmark"] if has_cf else [])

    return {"predicate_match_scores": scores, "strategy_recommendation": strategy, "key_gaps": gaps, "fastest_path": fastest}
