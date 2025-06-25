# Product Comparator with FAISS, spaCy NER, and LLM fallback

import re
import os
import requests
import numpy as np
from dotenv import load_dotenv
from rapidfuzz import fuzz
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer, util
import faiss
import spacy

# === Load spaCy NER model ===
ner_model = spacy.load("ner_model")

def extract_using_ner(text, field):
    doc = ner_model(text)
    for ent in doc.ents:
        if ent.label_ == field:
            return ent.text
    return None

# === Load API Key ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Embedding Model ===
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === FAISS Indexes for Grades and Standards ===
known_grades = ["Fe 500D", "Fe 550", "Fe 415", "Type I", "Type V", "Class I", "Class II"]
grade_embeddings = model.encode(["query: " + g for g in known_grades])
grade_index = faiss.IndexFlatL2(grade_embeddings.shape[1])
grade_index.add(np.array(grade_embeddings))

known_standards = ["IS 1786", "IS 456", "IS 10262", "IS 4031", "ASTM A706", "ASTM A615"]
standard_embeddings = model.encode(["query: " + s for s in known_standards])
standard_index = faiss.IndexFlatL2(standard_embeddings.shape[1])
standard_index.add(np.array(standard_embeddings))

# === Preprocessing ===
def preprocess(text):
    return text.lower().strip().replace("_", " ").replace(":-", ":").replace(";", "; ")

# === Matching Functions ===
def semantic_match(val1, val2):
    if not val1 or not val2:
        return False
    emb1 = model.encode(val1, convert_to_tensor=True)
    emb2 = model.encode(val2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    return similarity.item() > 0.85

def match_grade_faiss(text):
    if not text: return None
    input_embedding = model.encode(["query: " + text])
    D, I = grade_index.search(np.array(input_embedding), k=1)
    return known_grades[I[0][0]] if D[0][0] < 1.0 else None

def match_standard_faiss(text):
    if not text: return None
    input_embedding = model.encode(["query: " + text])
    D, I = standard_index.search(np.array(input_embedding), k=1)
    return known_standards[I[0][0]] if D[0][0] < 1.0 else None

# === LLM Fallback ===
def llm_extract_single_field(text, field):
    prompt = f"""
From the following product description, extract only the value of the field: "{field}".

Description:
{text}

Respond with a clean field value like "TMT", "Fe 500D", "IS 1786", or "Loose".
If not found, return exactly: Unknown. Do not add explanation or context.
"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            }
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except:
        return None

# === Field Extractors ===
def extract_grade(text):
    return extract_using_ner(text, "Grade") or match_grade_faiss(text)

def extract_diameter(text):
    match = re.search(r"(\d{1,3}\.?\d*)\s?mm", text)
    return f"{float(match.group(1)):.2f} mm" if match else None

def extract_length(text):
    match = re.search(r"(\d{4,5}\.?\d*)\s?mm", text)
    return f"{float(match.group(1)):.2f} mm" if match else None

def extract_standard(text):
    return extract_using_ner(text, "Standard") or match_standard_faiss(text)

def extract_material(text):
    result = extract_using_ner(text, "Material") or llm_extract_single_field(text, "Material")
    return None if result.lower() == "unknown" else result

def extract_form(text):
    result = extract_using_ner(text, "Form") or llm_extract_single_field(text, "Form")
    return None if result.lower() == "unknown" else result

# === Field Comparison ===
def compare_field(val1, val2):
    if not val1 and not val2:
        return ("⚪ Not Mentioned", "-", "-")
    elif val1 == val2:
        return ("✅ Exact Match", val1, val2)
    elif fuzz.ratio(val1, val2) > 85:
        return ("✅ Fuzzy Match", val1, val2)
    elif semantic_match(val1, val2):
        return ("✅ Semantic Match", val1, val2)
    else:
        return ("❌ Mismatch", val1 or "-", val2 or "-")

# === Pretty Report ===
def print_report(string1, string2, results):
    print("\nString 1:\n", string1)
    print("\nString 2:\n", string2)
    print("\n🔍 Comparison Report:\n")
    table = PrettyTable()
    table.field_names = ["Aspect", "String 1", "String 2", "Match Status"]
    for aspect, val1, val2, status in results:
        table.add_row([aspect, val1, val2, status])
    print(table)

# === Comparison Runner ===
def compare_strings(string1, string2):
    s1 = preprocess(string1)
    s2 = preprocess(string2)
    results = []
    aspects = [
        ("Grade", extract_grade),
        ("Diameter", extract_diameter),
        ("Material", extract_material),
        ("Form", extract_form),
        ("Length", extract_length),
        ("Standard", extract_standard),
    ]
    for aspect, func in aspects:
        val1 = func(s1)
        val2 = func(s2)
        status, v1, v2 = compare_field(val1, val2)
        results.append((aspect, v1, v2, status))
    print_report(string1, string2, results)

# === CLI Entry ===
if __name__ == "__main__":
    print("\n🔧 Product Comparator - Enter two descriptions\n")
    s1 = input("Enter String 1:\n")
    s2 = input("Enter String 2:\n")
    compare_strings(s1, s2)
