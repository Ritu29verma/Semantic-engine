import re
import requests
import json

def clean_text(text: str) -> str:
    """Remove links, ads, and extra spaces from input text."""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"(Follow Us|Download Notes|Telegram|Edushine Classes)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()

def _extract_json(raw: str):
    """Extract valid JSON array from model output string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON array in raw text
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def generate_flashcards(text: str, model: str = "llama3.1:8b", n_cards: int = 6, timeout: int = 120):
    cleaned_text = clean_text(text)

    prompt = f"""
You are an experienced university professor creating **exam-style flashcards**.  

### Rules:
- Produce EXACTLY {n_cards} flashcards.  
- Format: JSON list of objects → [{{"question": "...", "answer": "..."}}]  
- **NO "Key Point 1/2/3"**.  
- **NO copy-pasting text or URLs**.  
- Questions should test concepts (definitions, functions, comparisons, importance).  
- Answers should be short, clear, 1–3 sentences, factual.  

### Good Examples:
[
  {{"question": "What is the main function of the Application Layer in the OSI model?", 
    "answer": "It enables application-to-application communication by providing services like email, file transfer, and web browsing."}},
  {{"question": "How does the Application Layer differ from the Presentation Layer?", 
    "answer": "The Application Layer provides services to users, while the Presentation Layer ensures proper data formatting, encryption, and compression."}}
]

### Bad Examples (DO NOT DO THIS):
- "Key Point 1: Computer Network (BCS603)..."
- "Visit Telegram for notes..."
- "Unit 5 summary..."

Now, generate the {n_cards} flashcards based on this study material:  

{cleaned_text[:4000]}
"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0}
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        data = _extract_json(raw) if isinstance(raw, str) else raw

        if not isinstance(data, list):
            raise ValueError("Model did not return a JSON array.")

        cleaned = []
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                q = str(item["question"]).strip()
                a = str(item["answer"]).strip()
                if q and a:
                    cleaned.append({"question": q, "answer": a})

        if not cleaned:
            raise ValueError("No valid flashcards in JSON.")

        return cleaned

    except Exception as e:
        return [{"question": "Parsing error", "answer": f"Check model output. {e}"}]
