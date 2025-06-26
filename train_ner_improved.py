import spacy
from spacy.tokens import DocBin
import sys
import os
sys.path.append(".")
from train_split import TRAIN_DATA

# ---
# IMPROVED NER TRAINING SCRIPT
# This script provides multiple model options and better error handling
# ---

def check_model_availability():
    """Check which models are available and return the best option"""
    models_to_try = [
        "en_core_web_trf",  # Best accuracy (transformer)
        "en_core_web_lg",   # Good accuracy (large)
        "en_core_web_md",   # Medium accuracy
        "en_core_web_sm"    # Fastest (small)
    ]
    
    for model in models_to_try:
        try:
            nlp = spacy.load(model)
            print(f"✅ Found model: {model}")
            return model
        except OSError:
            print(f"❌ Model not found: {model}")
            continue
    
    print("⚠️  No pre-trained models found. Using blank English model.")
    return "blank"

def create_improved_config():
    """Create an improved training configuration"""
    config_content = """[paths]
train = "train_improved.spacy"
dev = "dev_improved.spacy"
vectors = null

[system]
gpu_allocator = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"

[training.batcher]
size = 1000
buffer = 2000

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 1e-8
learn_rate = 0.001

[training.dropout]
@architectures = "spacy.Dropout.v1"
rate = 0.1

[training.patience]
@schedules = "compounding.v1"
start = 1600
compound = 1.001
t = 0.0

[training.max_epochs]
@schedules = "constant.v1"
value = 100

[training.max_steps]
@schedules = "constant.v1"
value = 20000

[training.eval_frequency]
@schedules = "constant.v1"
value = 200

[training.accumulate_gradient]
@schedules = "constant.v1"
value = 3

[nlp]
lang = "en"
pipeline = ["ner"]
batch_size = 1000

[components]

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 128
maxout_pieces = 2
use_upper = true
nO = null
"""
    return config_content

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
    
    convert_to_spacy(train_data, "train_improved.spacy")
    convert_to_spacy(dev_data, "dev_improved.spacy")
    
    return len(train_data), len(dev_data)

def train_improved_model():
    """Train the improved NER model"""
    print("🚀 Starting improved NER training...")
    
    # Check available models
    model_name = check_model_availability()
    
    if model_name == "blank":
        nlp = spacy.blank("en")
    else:
        nlp = spacy.load(model_name)
    
    # Add NER component if not present
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add labels
    print("🏷️  Adding entity labels...")
    for _, annotations in TRAIN_DATA:
        entities = annotations.get("entities")
        if entities:
            for ent in entities:
                ner.add_label(ent[2])
    
    # Prepare training data
    train_count, dev_count = prepare_training_data()
    
    # Create config
    config_content = create_improved_config()
    
    # Save config
    os.makedirs("configs", exist_ok=True)
    with open("configs/config_improved.cfg", "w") as f:
        f.write(config_content)
    
    # Train using spaCy CLI
    import subprocess
    print(f"🎯 Training with {train_count} train and {dev_count} dev examples...")
    
    result = subprocess.run([
        "python", "-m", "spacy", "train", "configs/config_improved.cfg",
        "--output", "ner_model_improved",
        "--paths.train", "train_improved.spacy",
        "--paths.dev", "dev_improved.spacy",
        "--verbose"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        print("📁 Model saved to 'ner_model_improved/'")
    else:
        print("❌ Training failed:")
        print(result.stderr)
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Starting improved NER training pipeline...")
    success = train_improved_model()
    
    if success:
        print("\n🎉 Training completed! You can now use the improved model.")
        print("📊 To evaluate: python evaluate_improved.py")
    else:
        print("\n❌ Training failed. Check the error messages above.") 