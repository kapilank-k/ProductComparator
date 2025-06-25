# 🧠 Product Comparator with NER + FAISS + LLM Fallback

This project compares two unstructured product descriptions (like cement or steel bars) and extracts structured fields using:
- ✅ spaCy-trained NER model
- ✅ FAISS + SentenceTransformers for semantic similarity
- ✅ Groq-hosted LLM (Mixtral) as fallback extractor
- ✅ Pretty comparison table output

---

## 🚀 Features

- Extracts: `Grade`, `Form`, `Material`, `Standard`, `Length`, `Diameter`
- Custom NER trained on 2000+ real and noisy samples
- Semantic matching for `Grades`, `Standards`
- LLM fallback for unclear/missing fields
- Test & Evaluate with Precision, Recall, F1
- CLI tool to compare any 2 strings easily

---

## 🔧 Usage

### 1. Compare Two Product Descriptions
``bash
python comparator.py

2. Train the NER model
bash
Copy code

python convert.py  # convert to .spacy
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy


3. Evaluate on Test Set

python evaluate_on_test.py



4.Folder Structure 
.
├── comparator.py              # Main comparison logic
├── original_training_data.py # Base clean data
├── noisy_training_data.py    # 1000+ generated samples
├── base_training_data.py     # Combines both
├── test_data.py              # Final test set
├── convert.py                # Converts to .spacy format
├── config.cfg                # spaCy NER training config
├── evaluate_on_test.py       # Computes precision, recall, F1
├── output/                   # Trained NER model lives here
└── README.md                 # This file


Local dev setup 
# Step 1: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # For Windows

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run comparator
python comparator.py


Evaluation report generator
# Add this inside your evaluate_on_test.py at the end
with open("evaluation_report.txt", "w") as f:
    f.write("📊 Test Set Evaluation Report:\n")
    f.write(f"✅ Correct Predictions: {correct}\n")
    f.write(f"🔍 Total Predicted:      {len(pred_entities)}\n")
    f.write(f"🎯 Total Actual Labels:  {len(true_entities)}\n\n")
    f.write(f"📈 Precision: {precision:.2f}\n")
    f.write(f"📉 Recall:    {recall:.2f}\n")
    f.write(f"⭐ F1 Score:  {f1:.2f}\n")

