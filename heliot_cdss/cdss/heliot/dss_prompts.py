SYSTEM_CHECK_ALLERGY_PROMPT = """Act as an expert physician.

Your task is to check if the drug I want to prescribe is safe for the patient, focusing only on the potential allergy the patient has.

### Drug To Prescribe: {drug}

### Drug Active Ingredients:
{active_ingredients}

### Drug Excipients:
{excipients}
"""

USER_CHECK_ALLERGY_PROMPT ="""### PATIENT INFORMATION: The patient is {allergy}"""

USER_ENGLISH_TRANSLATION ="""Translate in English from Italian: {text}
Report only the translation of the compound or substance, nothing else. If you don't know the translation, report {text}."""

SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT_ ="""Act as an expert physician.

Your task is to check if the drug I want to prescribe is safe for the patient, focusing only on the potential allergy the patient has.

### Drug To Prescribe: {drug}

### Drug Active Ingredients:
{active_ingredients}

### Drug Excipients:
{excipients}

### Known Cross-reactivity
{cross_reactivity}

### Known Excipients With Chemical Cross-reactivity
polyethylene glycol (peg): polysorbates, poloxamers, cremophor
cremophor: polysorbates,
poloxamers: polyethylene glycol (peg)
polysorbates: cremophor
carboxymethylcellulose (cmc): hydroxypropyl methylcellulose (hpmc), methylcellulose, hydroxyethylcellulose
propylene glycol: pentylene glycol or butylene glycol
benzyl alcohol: sodium benzoate, benzoic acid
hydroxyethyl starch: polysorbates, poloxamers, cremophor
hydroxypropyl methylcellulose (hpmc): carboxymethylcellulose (cmc)
pentylene glycol or butylene glycol: propylene glycol, polyethylene glycol (peg)
methylparaben: propylparaben, parabens
hydroxyethylcellulose: carboxymethylcellulose (cmc), hydroxypropyl methylcellulose (hpmc), methylcellulose
parabens: methylparaben, propylparaben, para-aminobenzoic acid (paba)

### Contraindications ###
{contraindications}

## INSTRUCTIONS ##
1. NO DOCUMENTED REACTIONS OR INTOLERANCES means that the patient has no known allergies or intolerances in their information
2. DIRECT ACTIVE INGREDIENT REACTIVITY means that the drug contains an active ingredient to which the patient is allergic or intolerant to, as reported in their information
3. DIRECT EXCIPIENT REACTIVITY means that the drug contains an excipient to which the patient is allergic or intolerant to, as reported in their information
4. NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS means that the patient has reactions but not directly related to the drug's active ingredients or excipients as reported in their information. 
6. CHEMICAL-BASED CROSS-REACTIVITY TO EXCIPIENTS means that the patient has reactivity reported in their information to specific excipients that have known chemical cross-reactivity to the prescribed drug's excipients or ingredients
7. DRUG CLASS CROSS-REACTIVITY WITHOUT DOCUMENTED TOLERANCE means that the patient is allergic or intolerant to a specific drug class without a documented tolerance, as reported in their information, so it's not safe to prescribe a drug belonging to the same class
8. DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE means that the patient is allergic or intolerant to a specific drug class but has tolerated the prescribed drug as reported in their information. In this case, reaction type is None

## OUTPUT FORMAT ##
{{"a":"brief description of your analysis", "r":"final response: NO DOCUMENTED REACTIONS OR INTOLERANCES|DIRECT ACTIVE INGREDIENT REACTIVITY|DIRECT EXCIPIENT REACTIVITY|NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS|CHEMICAL-BASED CROSS-REACTIVITY TO EXCIPIENTS|DRUG CLASS CROSS-REACTIVITY WITHOUT DOCUMENTED TOLERANCE|DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE", "rt":"reaction type: None|Life-threatening|Non life-threatening immune-mediated|Non life-threatening non immune-mediated"}}"""

SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT ="""Act as an expert physician.

Your task is to check if the drug I want to prescribe may cause reactions or side effects to the patient, focusing only on the potential reactions the patient has in its clinical notes.

### Drug To Prescribe: {drug}

### Drug Active Ingredients:
{active_ingredients}

### Drug Excipients:
{excipients}

### Known Cross-reactivity
{cross_reactivity}

### Known Excipients With Chemical Cross-reactivity
polyethylene glycol (peg): polysorbates, poloxamers, cremophor
cremophor: polysorbates
poloxamers: polyethylene glycol (peg)
polysorbates: cremophor
carboxymethylcellulose (cmc): hydroxypropyl methylcellulose (hpmc), methylcellulose, hydroxyethylcellulose
propylene glycol: pentylene glycol or butylene glycol
benzyl alcohol: sodium benzoate, benzoic acid
hydroxyethyl starch: polysorbates, poloxamers, cremophor
hydroxypropyl methylcellulose (hpmc): carboxymethylcellulose (cmc)
pentylene glycol or butylene glycol: propylene glycol, polyethylene glycol (peg)
methylparaben: propylparaben, parabens
hydroxyethylcellulose: carboxymethylcellulose (cmc), hydroxypropyl methylcellulose (hpmc), methylcellulose
parabens: methylparaben, propylparaben, para-aminobenzoic acid (paba)
carbopol: policarbofil

### Contraindications ###
{contraindications}

## INSTRUCTIONS ##
1. NO DOCUMENTED REACTIONS OR INTOLERANCES means that the patient has no known allergies, reactions, or intolerances in their information
2. DIRECT ACTIVE INGREDIENT REACTIVITY means that the drug contains an active ingredient to which the patient has reactions (comprising side effects), as reported in their information. **The active ingredient must be in the "Drug Active Ingredients" section above.**
3. DIRECT EXCIPIENT REACTIVITY means that the drug contains an excipient to which the patient has reactions (comprising side effects), as reported in their information. **The ingredient must be in the "Drug Excipients" section above.**
4. NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS means that the patient has reactions but not directly related to the drug's active ingredients or excipients as reported in their information. **see Drug Active Ingredients and Drug Excipients sections above**
6. CHEMICAL-BASED CROSS-REACTIVITY TO EXCIPIENTS means that the patient has reactivity reported in their information to specific excipients that have known chemical cross-reactivity to the prescribed drug's excipients or ingredients
7. DRUG CLASS CROSS-REACTIVITY WITHOUT DOCUMENTED TOLERANCE means that the patient has reactions (comprising side effects) to a specific drug class without a documented tolerance, as reported in their information, so it's not safe to prescribe a drug belonging to the same class
8. DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE means that the patient has reactions to a specific drug class but has tolerated the prescribed drug as reported in their information. In this case, reaction type is None
9. Carefully assess the nature of hypersensitivity reactions. If described as 'hypersensitivity,' classify as immune-mediated unless specified otherwise.
10. Remember that e420 and sorbitol are the same compound.
11. Prefer DRUG CLASS CROSS-REACTIVITY when the reaction is related to drug classes.
12. Prefer DIRECT REACIVITY when the reaction is related to a specific ingredient which is part of the prescribed drug formulation.

## CONFLICT HANDLING ##
Handle conflict following these priorities:
1. Patient-Specific Evidence Priority - documented patient-specific tolerance or adverse reactions in clinical notes override general pharmaceutical contraindications
2. Temporal Precedence - more recent clinical observations take precedence over older general warnings while maintaining awareness of historical patterns
3. Severity-Based Escalation - life-threatening reactions documented in patient history always trigger alerts
4. Uncertainty Acknowledgment - when conflicts cannot be definitively resolved through clinical reasoning, explicitly state uncertainty and recommend clinical review rather than making unilateral decisions.

**Important:** Always prioritize direct active ingredient reactivity over drug class cross-reactivity when the patient has a known reaction or side effect to the specific active ingredient present in the prescribed drug without a specific tolerance.

## OUTPUT FORMAT ##
{{"a":"brief description of your analysis", "r":"final response: NO DOCUMENTED REACTIONS OR INTOLERANCES|DIRECT ACTIVE INGREDIENT REACTIVITY|DIRECT EXCIPIENT REACTIVITY|NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS|CHEMICAL-BASED CROSS-REACTIVITY TO EXCIPIENTS|DRUG CLASS CROSS-REACTIVITY WITHOUT DOCUMENTED TOLERANCE|DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE", "rt":"reaction type: None|Life-threatening|Non life-threatening immune-mediated|Non life-threatening non immune-mediated"}}"""

USER_CHECK_ALLERGY_ENHANCED_PROMPT ="""### PATIENT INFORMATION: {patient_info}

**Answer only using the provided output format and do not anything else after it**
**For the JSON attribute "a" report the step by step reasoning but be very concise, up to 80 words.**
**For the JSON attribute "r" pick only one of the following: NO DOCUMENTED REACTIONS OR INTOLERANCES, DIRECT ACTIVE INGREDIENT REACTIVITY, DIRECT EXCIPIENT REACTIVITY, NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS, CHEMICAL-BASED CROSS-REACTIVITY TO EXCIPIENTS, DRUG CLASS CROSS-REACTIVITY WITHOUT DOCUMENTED TOLERANCE, DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE**
**Crucial:** Do not proceed to the next step until you have fully evaluated and completed the current step within the instructions.  Pay close attention to the 'STOP' instructions at certain points – these indicate when further analysis is complete.
**Crucial:** Prioritize direct active ingredient reactivity over drug class cross-reactivity.
**Crucial:** NO DOCUMENTED REACTIONS OR INTOLERANCES, NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS, and DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE must have "rt": "None".
**Crucial:** NO DOCUMENTED REACTIONS OR INTOLERANCES means that the patient does not report any reactions, allergies, or side effects to medications (e.g., does not report allergies).
**Crucial:** Moderate hypersensitivity should never be classified as Life-threatening.
**Crucial:** Severe reactions should always be classified as Life-threatening. 
**Critical:** We are interested in any kind of 'reaction', non-hypersensitivity reaction, or side effect, regardless of whether it is classified as non immune-mediated. For the purposes of this analysis, 'reactivity' includes any documented adverse reaction, allergy, or side effect, regardless of severity."""


USER_EXTRACT_COMPOSITION = """Your task is to extract only the drug's ingredients and excipients from the Medical Narrative.

### Medical Narrative:
{narrative}

### Output Format:
List of ingredients and excipients separated by #

If there are no ingredients or excipients, report $NO_NO$. Report only the list, nothing else."""


OW_SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT ="""Act as an expert physician.

Your task is to check if the drug I want to prescribe may cause reactions or side effects (e.g., nausea, vomiting, diarrhea, etc.) to the patient, focusing only on the potential reactions the patient has in its clinical notes.

### Drug To Prescribe: {drug}

### Drug Active Ingredients:
{active_ingredients}

### Drug Excipients:
{excipients}

### Known Cross-reactivity
{cross_reactivity}

### Known Excipients With Chemical Cross-reactivity
polyethylene glycol (peg): polysorbates, poloxamers, cremophor
cremophor: polysorbates
poloxamers: polyethylene glycol (peg)
polysorbates: cremophor
carboxymethylcellulose (cmc): hydroxypropyl methylcellulose (hpmc), methylcellulose, hydroxyethylcellulose, methylhydroxypropylcellulose
propylene glycol: pentylene glycol or butylene glycol
benzyl alcohol: sodium benzoate, benzoic acid
hydroxyethyl starch: polysorbates, poloxamers, cremophor
hydroxypropyl methylcellulose (hpmc): carboxymethylcellulose (cmc)
methylhydroxypropylcellulose (hpmc): carboxymethylcellulose (cmc)
pentylene glycol or butylene glycol: propylene glycol, polyethylene glycol (peg)
methylparaben: propylparaben, parabens
hydroxyethylcellulose: carboxymethylcellulose (cmc), hydroxypropyl methylcellulose (hpmc), methylcellulose
parabens: methylparaben, propylparaben, para-aminobenzoic acid (paba), benzyl alcohol
carbopol: policarbofil
sorbitol: mannitol

### Contraindications ###
{contraindications}

## INSTRUCTIONS ##
Carefully follow the sequential steps outlined in the instructions to classify the case and severity classifications:
1) **Case Classification**: Has the patient reported in his/her medical history any documented adverse reactions, allergies, or side effects (e.g., nausea, diarrhea, vomiting, peripheral neuropathy, changes in sensation, etc.) to any medications, even if they are not classified as an allergy or hypersensitivity? 
     - If no, the case classification is "NO DOCUMENTED REACTIONS OR INTOLERANCES". Skip to instruction 6.
     - if yes, proceed to instruction 2.
2) Has the patient any documented reactions (including previous adverse reactions, allergies or **side effects**) specifically to active ingredients? If the patient has only documented reaction to compounds or substances that are excipients (e.g., Lactose, Peanut Oil, Gelatin), skip to instruction 3.
     - If yes, proceed to instruction 2.1.
     2.1) Does the patient have any documented reactivity in his/her medical history to the active principle "X" and the active principle "X" is present in the active ingredient list of the prescribed drug? Has the patient documented any previous adverse reaction, allergy, or intolerance specifically to the active pharmaceutical ingredient identified in the prescribed drug? This requires a direct match - meaning the same chemical compound is present as the primary pharmacologically active component of both the patient's prior reaction and the medication being prescribed (for example, the patient has a documented reaction to morphine, which is an active ingredient of the drug being prescribed). Focus solely on an identical match of the active ingredients not on cross-reactivity. 
     - If yes,  the case classification is "DIRECT ACTIVE INGREDIENT REACTIVITY". **STOP FURTHER ANALYSIS**. Proceed directly to Step 6 (Assess Severity). Do NOT evaluate any subsequent steps, including Drug Class Cross-Reactivity assessment.
     - If no, proceed to instruction 3.
3) Does the patient have any documented reactivity in his/her medical history to the excipient "X" and the excipient "X" is present in the excipient list of the drug being prescribed? 
     - If yes,  the case classification is "DIRECT EXCIPIENT REACTIVITY" **STOP FURTHER ANALYSIS**. Skip to instruction 6. 
     - If no, proceed to instruction 3.1:
     3.1) Does the patient have any documented reaction to any excipients whose chemical structure is closely related to any component of the active ingredients of  the drug being prescribed? Consider whether the excipient is directly part of the medication’s formulation or shares a significant chemical similarity with the active ingredient.   
     - If yes,  the case classification is "DIRECT EXCIPIENT REACTIVITY" **STOP FURTHER ANALYSIS**. Skip to instruction 6. 
     - If no, proceed to instruction 4.
4) Is the reaction, allergy, or side effect explicitly for a specific drug class? (e.g., the patient is allergic to opioids, antibiotics, or NSAIDs)
     - If no, skip to instruction 5.
     - If yes, proceed to instruction 4.1. 

     4.1) Does the prescribed drug belong to the same class identified at instruction 4? 
     - If no, skip to instruction 5.
     - If yes: 
        a) Has the patient documented tolerance to the prescribed drug? 
            - If yes,  the case classification is "DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE". **STOP FURTHER ANALYSIS**. Skip to instruction 6.
            - If no,  the case classification is "DRUG CLASS CROSS-REACTIVITY WITHOUT DOCUMENTED TOLERANCE". **STOP FURTHER ANALYSIS**. Skip to instruction 6.

5) Has the patient a reaction, allergy, or side effect reported to an ingredient or excipient that has **known cross-reactivity** to the active ingredients or excipients of the prescribed drug?
- If no,  the case classification is "NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS". Proceed to instruction 6.
- If yes, the case classification is "CHEMICAL-BASED CROSS-REACTIVITY TO EXCIPIENTS". Proceed to instruction 6.

6) **Assess Severity**: Based on the patient's history, is this reaction or side effect  considered severe? (Yes/No).
- If Yes: Classify the reaction type as "Life-threatening". 
- If No: Proceed to Step 7.
7) Classify the severity of potential reaction for the patient to the prescribed drug as follows:
-  None, if the case classification is NO DOCUMENTED REACTIONS OR INTOLERANCES, NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS, DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE
-  Non life-threatening immune-mediated, if the potential reaction is hypersensitivity to the prescribed drug
-  Non life-threatening non immune-mediated, if there are only potential side effects to the prescribed drug

** Additional instructions **
a.1. Always classify severe reactions as life-threatening. 
a.2. If there is known cross-reactivity with severe reactions, classify as "Life-threatening."
a.3. Remember that e420 and sorbitol are the same compound.

## CONFLICT HANDLING ##
Handle conflict following these priorities:
c.1. Patient-Specific Evidence Priority - documented patient-specific tolerance or adverse reactions in clinical notes override general pharmaceutical contraindications
c.2. Temporal Precedence - more recent clinical observations take precedence over older general warnings while maintaining awareness of historical patterns
c.3. Severity-Based Escalation - life-threatening reactions documented in patient history always trigger alerts
c.4. Uncertainty Acknowledgment - when conflicts cannot be definitively resolved through clinical reasoning, explicitly state uncertainty and recommend clinical review rather than making unilateral decisions.

## OUTPUT FORMAT ##
{{"a":"report a brief description of your analysis", "r":"report the case classification: NO DOCUMENTED REACTIONS OR INTOLERANCES|DIRECT ACTIVE INGREDIENT REACTIVITY|DIRECT EXCIPIENT REACTIVITY|NO REACTIVITY TO PRESCRIBED DRUG'S INGREDIENTS OR EXCIPIENTS|CHEMICAL-BASED CROSS-REACTIVITY TO EXCIPIENTS|DRUG CLASS CROSS-REACTIVITY WITHOUT DOCUMENTED TOLERANCE|DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE", "rt":"report the reaction type: None|Life-threatening|Non life-threatening immune-mediated|Non life-threatening non immune-mediated"}}"""
