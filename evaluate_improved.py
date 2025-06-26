import spacy
from collections import defaultdict
import sys
sys.path.append(".")
from test_split import TRAIN_DATA as test_data
from prettytable import PrettyTable
import json
import os

# ---
# IMPROVED EVALUATION SCRIPT
# Provides detailed metrics, error analysis, and actionable insights
# ---

def load_model(model_path="ner_model_improved"):
    """Load the trained model with error handling"""
    try:
        nlp = spacy.load(model_path)
        print(f"✅ Loaded model from: {model_path}")
        return nlp
    except OSError:
        print(f"❌ Model not found at: {model_path}")
        print("Trying alternative models...")
        
        # Try alternative models
        alternatives = ["ner_model", "model-blank"]
        for alt in alternatives:
            try:
                nlp = spacy.load(alt)
                print(f"✅ Loaded model from: {alt}")
                return nlp
            except OSError:
                continue
        
        print("❌ No models found. Please train a model first.")
        return None

def evaluate_ner_detailed(nlp, test_data):
    """Comprehensive NER evaluation with detailed metrics"""
    metrics = {}
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    error_examples = []
    
    print(f"🔍 Evaluating on {len(test_data)} test examples...")
    
    for i, (text, annotations) in enumerate(test_data):
        # Get true entities
        true_ents = set((start, end, label) for start, end, label in annotations.get("entities", []))
        
        # Get predicted entities
        doc = nlp(text)
        pred_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)
        
        # Calculate metrics per entity
        for start, end, label in true_ents:
            # Initialize metrics for this label if not exists
            if label not in metrics:
                metrics[label] = {"tp": 0, "fp": 0, "fn": 0, "examples": []}
            
            found = False
            for p_start, p_end, p_label in pred_ents:
                if (start, end) == (p_start, p_end):
                    if label == p_label:
                        metrics[label]["tp"] += 1
                    else:
                        metrics[label]["fn"] += 1
                        # Initialize metrics for predicted label if not exists
                        if p_label not in metrics:
                            metrics[p_label] = {"tp": 0, "fp": 0, "fn": 0, "examples": []}
                        metrics[p_label]["fp"] += 1
                        confusion_matrix[label][p_label] += 1
                        error_examples.append({
                            "text": text,
                            "true": (start, end, label),
                            "pred": (p_start, p_end, p_label),
                            "error_type": "misclassification"
                        })
                    found = True
                    break
            
            if not found:
                metrics[label]["fn"] += 1
                confusion_matrix[label]["MISSED"] += 1
                error_examples.append({
                    "text": text,
                    "true": (start, end, label),
                    "pred": None,
                    "error_type": "missed"
                })
        
        # Count remaining false positives
        for p_start, p_end, p_label in pred_ents:
            if not any((start, end) == (p_start, p_end) for start, end, _ in true_ents):
                # Initialize metrics for predicted label if not exists
                if p_label not in metrics:
                    metrics[p_label] = {"tp": 0, "fp": 0, "fn": 0, "examples": []}
                metrics[p_label]["fp"] += 1
                confusion_matrix["NONE"][p_label] += 1
                error_examples.append({
                    "text": text,
                    "true": None,
                    "pred": (p_start, p_end, p_label),
                    "error_type": "false_positive"
                })
        
        # Store examples for each entity type
        for label in metrics:
            if any(label == ent[2] for ent in true_ents):
                metrics[label]["examples"].append(text)
    
    return metrics, confusion_matrix, error_examples

def print_detailed_metrics(metrics, confusion_matrix, error_examples):
    """Print comprehensive evaluation results"""
    
    # Overall metrics
    total_tp = sum(m["tp"] for m in metrics.values())
    total_fp = sum(m["fp"] for m in metrics.values())
    total_fn = sum(m["fn"] for m in metrics.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("📊 OVERALL EVALUATION RESULTS")
    print("="*60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Total Entities: {total_tp + total_fn}")
    print(f"Correct Predictions: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    
    # Per-entity metrics
    print("\n" + "="*60)
    print("📈 PER-ENTITY PERFORMANCE")
    print("="*60)
    
    table = PrettyTable()
    table.field_names = ["Entity", "Precision", "Recall", "F1", "Support", "Errors"]
    table.align = "l"
    
    for entity, m in sorted(metrics.items()):
        ent_precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        ent_recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall) if (ent_precision + ent_recall) > 0 else 0
        support = m["tp"] + m["fn"]
        errors = m["fp"] + m["fn"]
        
        table.add_row([
            entity,
            f"{ent_precision:.3f}",
            f"{ent_recall:.3f}",
            f"{ent_f1:.3f}",
            support,
            errors
        ])
    
    print(table)
    
    # Error analysis
    print("\n" + "="*60)
    print("🔍 ERROR ANALYSIS")
    print("="*60)
    
    error_types = defaultdict(int)
    for error in error_examples:
        error_types[error["error_type"]] += 1
    
    print("Error Distribution:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    # Show some example errors
    print(f"\nSample Errors (showing first 5):")
    for i, error in enumerate(error_examples[:5]):
        print(f"\nError {i+1} ({error['error_type']}):")
        print(f"  Text: {error['text'][:100]}...")
        if error['true']:
            print(f"  True: {error['true']}")
        if error['pred']:
            print(f"  Pred: {error['pred']}")
    
    # Save detailed results
    save_results(metrics, confusion_matrix, error_examples)

def save_results(metrics, confusion_matrix, error_examples):
    """Save detailed results to files"""
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Save metrics
    metrics_dict = {}
    for entity, m in metrics.items():
        metrics_dict[entity] = {
            "tp": m["tp"],
            "fp": m["fp"], 
            "fn": m["fn"],
            "precision": m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0,
            "recall": m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0,
            "support": m["tp"] + m["fn"]
        }
    
    with open("evaluation_results/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save error examples
    with open("evaluation_results/error_examples.json", "w") as f:
        json.dump(error_examples, f, indent=2)
    
    print(f"\n💾 Results saved to 'evaluation_results/' directory")

def generate_improvement_suggestions(metrics, error_examples):
    """Generate actionable improvement suggestions"""
    print("\n" + "="*60)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*60)
    
    # Find worst performing entities
    worst_entities = []
    for entity, m in metrics.items():
        if m["tp"] + m["fn"] > 0:  # Only consider entities with examples
            f1 = 2 * (m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0) * (m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0) / ((m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0) + (m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0)) if ((m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0) + (m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0)) > 0 else 0
            worst_entities.append((entity, f1, m["tp"] + m["fn"]))
    
    worst_entities.sort(key=lambda x: x[1])
    
    print("Priority improvements based on performance:")
    for entity, f1, support in worst_entities[:3]:
        print(f"  🔴 {entity}: F1={f1:.3f}, Support={support}")
        print(f"     → Add more training examples for this entity")
    
    # Analyze error patterns
    missed_count = sum(1 for e in error_examples if e["error_type"] == "missed")
    fp_count = sum(1 for e in error_examples if e["error_type"] == "false_positive")
    misclass_count = sum(1 for e in error_examples if e["error_type"] == "misclassification")
    
    print(f"\nError pattern analysis:")
    print(f"  Missed entities: {missed_count} (improve recall)")
    print(f"  False positives: {fp_count} (improve precision)")
    print(f"  Misclassifications: {misclass_count} (improve entity boundaries)")
    
    if missed_count > fp_count:
        print("  → Focus on improving recall (add more diverse examples)")
    else:
        print("  → Focus on improving precision (add negative examples)")

if __name__ == "__main__":
    print("🔍 Starting improved NER evaluation...")
    
    # Load model
    nlp = load_model()
    if nlp is None:
        exit(1)
    
    # Evaluate
    metrics, confusion_matrix, error_examples = evaluate_ner_detailed(nlp, test_data)
    
    # Print results
    print_detailed_metrics(metrics, confusion_matrix, error_examples)
    
    # Generate suggestions
    generate_improvement_suggestions(metrics, error_examples)
    
    print("\n✅ Evaluation completed!") 