from typing import Dict

EXPLANATIONS = {
    "nv": "Melanocytic nevi (nv) are common benign moles. Generally low risk but monitor for changes.",
    "mel": "Melanoma is a serious skin cancer. High severity — seek a dermatologist promptly.",
    "bcc": "Basal cell carcinoma grows slowly. Medium severity — schedule a professional evaluation.",
    "akiec": "Actinic keratoses / Bowen's disease (akiec) may progress to cancer. High severity — see a specialist.",
    "bkl": "Benign keratosis-like lesions are non-cancerous. Low severity.",
    "df": "Dermatofibroma is a benign lesion. Low severity.",
    "vasc": "Vascular lesions (e.g., angiomas). Low severity.",
    "no_lesion": "No lesion detected. Routine self-checks are recommended.",
    "wrong": "The image does not appear to be a skin lesion. Try retaking a clear, close-up photo in good lighting.",
}

NEXT_STEPS = {
    "HIGH": "This may require urgent attention. Please book a dermatologist appointment as soon as possible.",
    "MEDIUM": "Professional evaluation recommended. Schedule a visit with a dermatologist.",
    "LOW": "Monitor the lesion for changes in size, shape, or color.",
    "NONE": "No immediate action needed. Maintain routine self-checks.",
}


def generate_response(user_message: str, context: Dict) -> str:
    user_message_lower = user_message.strip().lower()

    predicted_class = context.get("class", "")
    severity = context.get("severity", "NONE")
    is_wrong = context.get("is_wrong_image", False)

    if is_wrong:
        return EXPLANATIONS["wrong"]

    if any(k in user_message_lower for k in ["what is", "explain", "about", "info", "information"]):
        key = predicted_class if predicted_class in EXPLANATIONS else "no_lesion"
        return EXPLANATIONS[key]

    if any(k in user_message_lower for k in ["severity", "serious", "danger", "risk"]):
        return f"Severity: {severity}. {NEXT_STEPS.get(severity, '')}"

    if any(k in user_message_lower for k in ["save", "report", "download", "export"]):
        return "Use the 'Download Report' button to save a PDF or CSV containing the prediction details."

    if any(k in user_message_lower for k in ["hello", "hi", "help", "start"]):
        return "Hi! I can explain lesion types, guide next steps based on severity, and help you download reports. Ask me anything."

    # Default guidance
    if predicted_class:
        base = EXPLANATIONS.get(predicted_class, "")
        steps = NEXT_STEPS.get(severity, "")
        return f"Predicted: {predicted_class}. Severity: {severity}. {base} {steps}".strip()

    return "I can help with lesion information, severity guidance, and saving reports."
