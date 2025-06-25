import spacy

# Load your trained model
nlp = spacy.load("ner_model")

# Sample product descriptions to test
samples = [
    "TMT Fe500D bar 12mm IS 1786",
    "Loose OPC 53 bulk cement IS12269",
    "PC Strand CLASS II 12.5mm IS14268",
    "Fe415 TMT steel 12mm IS 1786",
]

for text in samples:
    doc = nlp(text)
    print(f"\n🔍 Text: {text}")
    for ent in doc.ents:
        print(f"  ➤ {ent.label_}: {ent.text}")
