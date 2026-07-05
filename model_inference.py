import torch
import os

# Global model, tokenizer, and engine state cache
_model = None
_tokenizer = None
_engine_name = None

def generate_response(user_symptoms: str, grounded_evidence: str) -> str:
    """
    Three-Tier Clinical Triage Generator:
    Tier 1: Fast Unsloth (requires active CUDA + unsloth library)
    Tier 2: Standard HF Transformers (compatible with CPU or non-unsloth setups)
    Tier 3: Failsafe Deterministic Rule-Based Triage (instant, no setup requirements)
    """
    global _model, _tokenizer, _engine_name

    # --- Tier 1: Unsloth Model Generation (CUDA Only) ---
    if torch.cuda.is_available():
        try:
            from unsloth import FastLanguageModel # type: ignore
            
            if _model is None or _engine_name != "Unsloth":
                max_seq_length = 2048
                dtype = None
                load_in_4bit = True
                
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="unsloth/Llama-3.2-1B-Instruct",
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                )
                FastLanguageModel.for_inference(model)
                _model = model
                _tokenizer = tokenizer
                _engine_name = "Unsloth"
                
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert clinical triage AI assistant. Analyze the patient case details "
                        "under the guidance of the provided Clinical Guidelines Reference. "
                        "Classify the urgency (Emergent, Urgent, Semi-Urgent, or Non-Urgent), justify your decision "
                        "with clear reasoning, and provide recommended next steps."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Clinical Guidelines Reference:\n{grounded_evidence}\n\n"
                        f"Patient Case Details:\n{user_symptoms}\n\n"
                        "Provide a structured response: Triage Category, Reasoning, and Immediate Next Steps."
                    )
                }
            ]

            inputs = _tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            outputs = _model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=0.1,
                top_p=0.9
            )

            input_length = inputs.shape[1]
            response = _tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            return f"{response}\n\n---\n*⚡ Powered by Unsloth (4-bit Llama-3.2-1B-Instruct on CUDA GPU)*"
            
        except Exception as e:
            # Log error and fall through to standard transformers
            print(f"Tier 1 (Unsloth) load failed: {str(e)}. Attempting Tier 2 fallback...")

    # --- Tier 2: Standard Hugging Face Transformers (CPU / GPU Fallback) ---
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if _model is None or _engine_name != "Transformers":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
            
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    "unsloth/Llama-3.2-1B-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    "unsloth/Llama-3.2-1B-Instruct",
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            
            _model = model
            _tokenizer = tokenizer
            _engine_name = "Transformers"
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert clinical triage AI assistant. Analyze the patient case details "
                    "under the guidance of the provided Clinical Guidelines Reference. "
                    "Classify the urgency (Emergent, Urgent, Semi-Urgent, or Non-Urgent), justify your decision "
                    "with clear reasoning, and provide recommended next steps."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Clinical Guidelines Reference:\n{grounded_evidence}\n\n"
                    f"Patient Case Details:\n{user_symptoms}\n\n"
                    "Provide a structured response: Triage Category, Reasoning, and Immediate Next Steps."
                )
            }
        ]

        inputs = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        outputs = _model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.1,
            top_p=0.9
        )

        input_length = inputs.shape[1]
        response = _tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        device_label = "CUDA GPU" if device == "cuda" else "CPU"
        return f"{response}\n\n---\n*🔄 Powered by standard HF Transformers (Llama-3.2-1B-Instruct on {device_label})*"

    except Exception as e:
        print(f"Tier 2 (Transformers) load failed: {str(e)}. Reverting to Tier 3 Failsafe Triage...")

    # --- Tier 3: Failsafe Deterministic Rule-Based Triage (Fallback) ---
    urgency = "Non-Urgent"
    reasoning = "General symptoms presented do not match any high-severity clinical rules in the reference library. The patient is advised to self-monitor and follow up with a primary care provider."
    actions = "- Monitor temperature and vital signs.\n- Schedule a regular primary care appointment if symptoms persist.\n- Seek immediate emergency care if new symptoms develop (e.g., chest pain, shortness of breath)."

    symptoms_lower = user_symptoms.lower()
    
    if "appendicitis" in grounded_evidence.lower() or "lower right abdominal" in symptoms_lower or "right lower quadrant" in symptoms_lower:
        urgency = "Emergent"
        reasoning = (
            "The patient presents with symptoms highly suggestive of acute appendicitis (e.g., lower right quadrant abdominal pain). "
            "Under the reference guidelines, this represents a surgical emergency with a high risk of perforation."
        )
        actions = (
            "- **Transport immediately to the nearest Emergency Department (ED)**.\n"
            "- Strictly **NPO** (Nothing by Mouth); do not administer food, water, or oral pain medications.\n"
            "- Monitor vitals closely during transport."
        )
    elif "cholangiocarcinoma" in grounded_evidence.lower() or "jaundice" in symptoms_lower or "bile duct" in symptoms_lower:
        urgency = "Urgent"
        reasoning = (
            "The symptoms (jaundice, potential biliary obstruction indications) suggest significant biliary pathology. "
            "While not immediately life-threatening within minutes, it requires rapid specialist diagnostic workup."
        )
        actions = (
            "- Schedule an **immediate gastroenterology or oncological evaluation** (within 24-48 hours).\n"
            "- Obtain hepatic panel and biliary imaging (ultrasound/MRCP).\n"
            "- Report to emergency services if accompanied by high fever or severe right upper quadrant pain (ascending cholangitis suspect)."
        )
    elif "common_cold" in grounded_evidence.lower() or "sore throat" in symptoms_lower or "sneezing" in symptoms_lower:
        urgency = "Non-Urgent"
        reasoning = (
            "Symptoms align with typical upper respiratory tract viral infection (common cold). "
            "There are no red flags or indications of lower airway involvement or airway compromise."
        )
        actions = (
            "- Supportive care including rest and adequate oral hydration.\n"
            "- Symptomatic treatment with over-the-counter antipyretics or throat lozenges.\n"
            "- Follow up if symptoms do not improve after 7-10 days, or worsen."
        )

    response = (
        f"### 📋 Failsafe Clinical Triage Report\n\n"
        f"**TRIAGE URGENCY CATEGORY**: {urgency}\n\n"
        f"**CLINICAL REASONING**:\n{reasoning}\n\n"
        f"**RECOMMENDED IMMEDIATE ACTIONS**:\n{actions}"
    )
    return f"{response}\n\n---\n*🛡️ Running in Failsafe Local Rule-Based Mode (No AI models loaded due to environment limitations)*"
