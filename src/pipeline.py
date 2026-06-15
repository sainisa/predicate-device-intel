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
    if use_demo or not api_key:
        return DEMO_COMPARISON
    # Live path would aggregate extracted intelligence and call comparison prompt
    return DEMO_COMPARISON


def match_new_device(device_description: str, api_key: str = None, use_demo: bool = True) -> dict:
    if use_demo or not api_key:
        return DEMO_MATCH
    return DEMO_MATCH
