# Factual Grounding Database (Source of Truth)
MEDICAL_KNOWLEDGE_BASE = {
    "appendicitis": (
        "Symptoms: Lower right abdominal pain, sudden onset, nausea, low-grade fever. "
        "Action: Suspect acute appendicitis. Requires immediate emergency surgical triage "
        "and evaluation. Do not delay or administer oral medication."
    ),
    "cholangiocarcinoma": (
        "Symptoms: Jaundice, clay-colored stools, dark urine, upper right abdominal pain, "
        "weight loss. Action: Schedule immediate oncological evaluation for bile duct pathologies."
    ),
    "common_cold": (
        "Symptoms: Rhinorrhea, mild sore throat, low-grade cough, sneezing. "
        "Action: Recommend symptomatic supportive care, hydration, and rest."
    )
}

def retrieve_medical_fact(user_symptoms: str) -> str:
    """
    Scans the local knowledge database for matching medical keywords 
    to retrieve grounded context for prompt injection.
    """
    query = user_symptoms.lower()
    
    if "lower right abdominal" in query or "right lower quadrant" in query:
        return MEDICAL_KNOWLEDGE_BASE["appendicitis"]
    elif "bile duct" in query or "jaundice" in query:
        return MEDICAL_KNOWLEDGE_BASE["cholangiocarcinoma"]
    elif "sore throat" in query or "sneezing" in query:
        return MEDICAL_KNOWLEDGE_BASE["common_cold"]
        
    return "No explicit matching reference found in the immediate local knowledge base. Proceed with standard general triage caution."
