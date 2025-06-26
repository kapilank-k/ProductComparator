import spacy
from spacy.cli.train import train
import os
from pathlib import Path
import sys
sys.path.append(".")
from train_split import TRAIN_DATA
from test_split import TRAIN_DATA as test_data

def create_config():
    config = {
        "paths": {
            "train": "train_split.py",
            "dev": "dev_split.py",
            "vectors": None
        },
        "system": {
            "gpu_allocator": None
        },
        "corpora": {
            "train": {
                "path": "train_split.py",
                "max_length": 0,
                "limit": 0
            },
            "dev": {
                "path": "dev_split.py",
                "max_length": 0,
                "limit": 0
            }
        },
        "training": {
            "dev_corpus": "corpora.dev",
            "train_corpus": "corpora.train",
            "optimizer": {
                "learn_rate": 0.001
            },
            "batcher": {
                "size": 128,
                "buffer": 256
            },
            "logger": {
                "path": "logs",
                "stdout": True
            },
            "optimizer": {
                "@optimizers": "Adam.v1",
                "beta1": 0.9,
                "beta2": 0.999,
                "L2_is_weight_decay": True,
                "L2": 0.01,
                "grad_clip": 1.0,
                "use_averages": False,
                "eps": 1e-8
            },
            "dropout": 0.2,
            "patience": 1600,
            "max_epochs": 50,
            "max_steps": 20000,
            "eval_frequency": 200
        },
        "nlp": {
            "lang": "en",
            "pipeline": ["ner"],
            "batch_size": 128
        },
        "components": {
            "ner": {
                "factory": "ner",
                "model": {
                    "@architectures": "spacy.TransitionBasedParser.v2",
                    "state_type": "ner",
                    "extra_state_tokens": False,
                    "hidden_width": 64,
                    "maxout_pieces": 2,
                    "use_upper": True,
                    "nO": None
                }
            }
        }
    }
    return config

def setup_directories():
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    # Save config
    import srsly
    config = create_config()
    srsly.write_json("configs/config.cfg", config)

def train_model():
    # Create blank English model
    nlp = spacy.blank("en")
    
    # Add NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    
    # Load training data to get labels
    from train_split import TRAIN_DATA
    
    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Save the blank model
    nlp.to_disk("model-blank")
    
    # Train using the config
    train(
        "configs/config.cfg",
        output_path="ner_model",
        overrides={"paths.train": "train_split.py", "paths.dev": "dev_split.py"}
    )

if __name__ == "__main__":
    print("🔧 Setting up training environment...")
    setup_directories()
    
    print("🚀 Starting model training...")
    train_model()
    
    print("✅ Training complete! Model saved to 'ner_model/'")
