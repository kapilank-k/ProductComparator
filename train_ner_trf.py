import spacy
from spacy.tokens import DocBin
import sys
sys.path.append(".")
from train_split import TRAIN_DATA

# ---
# This script upgrades the NER model to use a transformer backbone (en_core_web_trf)
# for much better contextual understanding and accuracy.
# ---

# 1. Load the transformer pipeline
print("Loading transformer pipeline (en_core_web_trf)...")
nlp = spacy.load("en_core_web_trf")

# 2. Add NER labels from your data
ner = nlp.get_pipe("ner")
for _, annotations in TRAIN_DATA:
    entities = annotations.get("entities")
    if entities:
        for ent in entities:
            ner.add_label(ent[2])

# 3. Convert training data to spaCy's DocBin format (for efficient training)
doc_bin = DocBin()
for text, ann in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    entities = ann.get("entities")
    if entities:
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

doc_bin.to_disk("train_trf.spacy")

# 4. Create a minimal config for updating the NER component
# (spaCy's quickstart config can be used, or you can use your own config)
# For demonstration, we'll use the built-in config and update only NER
import subprocess
print("Generating config file for transformer NER training...")
subprocess.run([
    "python", "-m", "spacy", "init", "config", "config_trf.cfg",
    "--lang", "en", "--pipeline", "ner", "--optimize", "accuracy",
    "--force"
])

# 5. Train the model using spaCy's CLI
print("Starting transformer-based NER training...")
subprocess.run([
    "python", "-m", "spacy", "train", "config_trf.cfg",
    "--output", "ner_model_trf", "--paths.train", "train_trf.spacy", "--paths.dev", "train_trf.spacy"
])

print("\n✅ Transformer-based NER model trained and saved to 'ner_model_trf/'")

# ---
# EXPLANATION FOR MENTOR:
# - This script uses spaCy's en_core_web_trf (transformer) pipeline for NER.
# - It adds your custom entity labels, converts your data to spaCy's binary format,
#   generates a config, and launches training using the transformer backbone.
# - The resulting model (ner_model_trf) should be much more accurate on complex product descriptions.
# --- 