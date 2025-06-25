import spacy
from spacy.training.example import Example
from training_data import TRAIN_DATA

# Step 1: Create blank English NLP pipeline
nlp = spacy.blank("en")

# Step 2: Add NER to pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Step 3: Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Step 4: Training loop
nlp.begin_training()
for i in range(30):  # Train for 30 epochs
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example])

# Step 5: Save model
nlp.to_disk("ner_model")
print("✅ Model trained and saved to 'ner_model/'")
