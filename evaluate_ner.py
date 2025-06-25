import spacy
from base_training_data import TRAIN_DATA

# Load trained spaCy model
nlp = spacy.load("ner_model")

# Initialize counters
correct = 0
predicted = 0
actual = 0

for text, annotations in TRAIN_DATA:
    true_ents = set((start, end, label) for start, end, label in annotations["entities"])
    doc = nlp(text)
    pred_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)

    correct += len(true_ents & pred_ents)
    predicted += len(pred_ents)
    actual += len(true_ents)

# Calculate precision, recall, F1
precision = correct / predicted if predicted > 0 else 0.0
recall = correct / actual if actual > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print("\n📊 NER Evaluation Results:")
print(f"✅ Correct Predictions: {correct}")
print(f"🔍 Total Predicted:      {predicted}")
print(f"🎯 Total Actual Labels:  {actual}")
print(f"\n📈 Precision: {precision:.2f}")
print(f"📉 Recall:    {recall:.2f}")
print(f"⭐ F1 Score:  {f1:.2f}")
