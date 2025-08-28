import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import sys
import os
sys.path.append(".")
from train_split import TRAIN_DATA

# ---
# SIMPLE BUT EFFECTIVE NER TRAINING
# Uses spaCy's basic training loop without complex configs
# ---

def filter_overlapping_entities(entities):
    """Filter out overlapping entities, keeping the longest one"""
    if not entities:
        return []
    
    # Sort by start position, then by length (longest first)
    sorted_entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    
    filtered = []
    for entity in sorted_entities:
        start, end, label = entity
        
        # Check if this entity overlaps with any existing entity
        overlaps = False
        for existing in filtered:
            ex_start, ex_end, _ = existing
            if (start < ex_end and end > ex_start):
                overlaps = True
                break
        
        if not overlaps:
            filtered.append(entity)
    
    return filtered

def prepare_training_data():
    """Prepare training data in spaCy format"""
    print("📊 Preparing training data...")
    
    # Split data into train/dev
    from split_data import stratified_split
    train_data, dev_data, _ = stratified_split(TRAIN_DATA, train_ratio=0.8, dev_ratio=0.2)
    
    # Convert to spaCy format
    def convert_to_spacy(data, filename):
        doc_bin = DocBin()
        for text, ann in data:
            doc = spacy.blank("en")(text)
            entities = ann.get("entities", [])
            
            # Filter overlapping entities
            filtered_entities = filter_overlapping_entities(entities)
            
            ents = []
            for start, end, label in filtered_entities:
                # Ensure valid span boundaries
                if start >= 0 and end <= len(text) and start < end:
                    span = doc.char_span(start, end, label=label)
                    if span is not None:
                        ents.append(span)
            
            doc.ents = ents
            doc_bin.add(doc)
        doc_bin.to_disk(filename)
        print(f"✅ Saved {len(data)} examples to {filename}")
    
    convert_to_spacy(train_data, "train_simple.spacy")
    convert_to_spacy(dev_data, "dev_simple.spacy")
    
    return train_data, dev_data

def train_simple_model():
    """Train using spaCy's simple training approach"""
    print("🚀 Starting simple NER training...")
    
    # Use blank model to avoid lookups issues
    print("✅ Using blank English model")
    nlp = spacy.blank("en")
    
    # Add NER component
    ner = nlp.add_pipe("ner")
    
    # Add labels
    print("🏷️  Adding entity labels...")
    for _, annotations in TRAIN_DATA:
        entities = annotations.get("entities")
        if entities:
            for ent in entities:
                ner.add_label(ent[2])
    
    # Prepare training data
    train_data, dev_data = prepare_training_data()
    
    # Start training
    nlp.begin_training()
    
    print(f"🎯 Training with {len(train_data)} examples...")
    
    # Training loop
    for epoch in range(30):  # Train for 30 epochs
        losses = {}
        
        # Shuffle training data
        import random
        random.shuffle(train_data)
        
        # Process training examples
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.1, losses=losses)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/30, Losses: {losses}")
    
    # Save the trained model
    nlp.to_disk("ner_model_simple")
    print("✅ Model saved to 'ner_model_simple/'")
    
    return True

def evaluate_simple_model():
    """Quick evaluation of the trained model"""
    print("\n🔍 Evaluating simple model...")
    
    try:
        nlp = spacy.load("ner_model_simple")
        
        # Test on a few examples
        test_examples = [
            "TMT Fe500D 12mm 12000mm IS 1786",
            "OPC 43 Grade Cement 50kg Bag",
            "PC Strand 12.5mm 16000mm IS 14268"
        ]
        
        for text in test_examples:
            doc = nlp(text)
            print(f"\nText: {text}")
            print("Entities found:")
            for ent in doc.ents:
                print(f"  {ent.text} -> {ent.label_}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Starting simple NER training pipeline...")
    
    # Train the model
    success = train_simple_model()
    
    if success:
        print("\n🎉 Training completed successfully!")
        
        # Quick evaluation
        evaluate_simple_model()
        
        print("\n✅ You can now use the simple model:")
        print("   - Model location: ner_model_simple/")
        print("   - To evaluate: python evaluate_improved.py")
        print("   - To compare products: python product_comparator_enhanced.py")
    else:
        print("\n❌ Training failed. Check the error messages above.") 