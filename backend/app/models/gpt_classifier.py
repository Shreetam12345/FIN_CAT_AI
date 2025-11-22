import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- RAW TEXT CLASSIFICATION ---------
def gpt_predict_texts(texts: list[str]) -> dict:
    prompt = (
        "Classify the following financial transaction text into a category like "
        "'Food', 'Bills', 'Shopping', 'Travel', etc.\n\n"
        f"Text: {texts[0]}\nCategory:"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )

    category = response.choices[0].message["content"].strip()
    return {"input": texts[0], "category": category}


# -------- EXCEL UPLOAD CLASSIFICATION (SIMPLE) ---------
def gpt_predict_excel(file_path: str) -> dict:
    return {"message": "Excel classification not implemented yet"}
