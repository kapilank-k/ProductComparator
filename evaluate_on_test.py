import spacy
from test_data import TEST_DATA

nlp = spacy.load("ner_model")

correct = 0
predicted = 0
actual = 0

for text, annotations in TEST_DATA:
    true_ents = set((start, end, label) for start, end, label in annotations["entities"])
    doc = nlp(text)
    pred_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)

    correct += len(true_ents & pred_ents)
    predicted += len(pred_ents)
    actual += len(true_ents)

precision = correct / predicted if predicted else 0.0
recall = correct / actual if actual else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print("\n📊 Test Set Evaluation:")
print(f"✅ Correct Predictions: {correct}")
print(f"🔍 Total Predicted:      {predicted}")
print(f"🎯 Total Actual Labels:  {actual}")
print(f"\n📈 Precision: {precision:.2f}")
print(f"📉 Recall:    {recall:.2f}")
print(f"⭐ F1 Score:  {f1:.2f}")
