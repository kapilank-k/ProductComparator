import spacy
from collections import defaultdict
from test_split import TRAIN_DATA as test_data
from prettytable import PrettyTable

def evaluate_ner(nlp, test_data):
    # Initialize entity-wise metrics
    metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for text, annotations in test_data:
        # Get true entities
        true_ents = set((start, end, label) for start, end, label in annotations["entities"])
        
        # Get predicted entities
        doc = nlp(text)
        pred_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)
        
        # Calculate true positives, false positives, and false negatives per entity
        for start, end, label in true_ents:
            found = False
            for p_start, p_end, p_label in pred_ents:
                if (start, end) == (p_start, p_end):
                    if label == p_label:
                        metrics[label]["tp"] += 1
                    else:
                        metrics[label]["fn"] += 1
                        metrics[p_label]["fp"] += 1
                        confusion_matrix[label][p_label] += 1
                    found = True
                    break
            if not found:
                metrics[label]["fn"] += 1
                confusion_matrix[label]["MISSED"] += 1
        
        # Count remaining false positives
        for p_start, p_end, p_label in pred_ents:
            if not any((start, end) == (p_start, p_end) for start, end, _ in true_ents):
                metrics[p_label]["fp"] += 1
                confusion_matrix["NONE"][p_label] += 1
    
    return metrics, confusion_matrix

def print_metrics(metrics, confusion_matrix):
    # Print overall metrics
    total_tp = sum(m["tp"] for m in metrics.values())
    total_fp = sum(m["fp"] for m in metrics.values())
    total_fn = sum(m["fn"] for m in metrics.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n📊 Overall Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    
    # Print per-entity metrics
    print("\n📈 Per-Entity Metrics:")
    table = PrettyTable()
    table.field_names = ["Entity", "Precision", "Recall", "F1", "Support"]
    
    for entity, m in sorted(metrics.items()):
        ent_precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        ent_recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall) if (ent_precision + ent_recall) > 0 else 0
        support = m["tp"] + m["fn"]
        
        table.add_row([
            entity,
            f"{ent_precision:.3f}",
            f"{ent_recall:.3f}",
            f"{ent_f1:.3f}",
            support
        ])
    
    print(table)
    
    # Print confusion matrix
    print("\n🔄 Confusion Matrix:")
    all_labels = sorted(set(
        list(confusion_matrix.keys()) +
        [l for rows in confusion_matrix.values() for l in rows.keys()]
    ))
    
    matrix_table = PrettyTable()
    matrix_table.field_names = ["True↓/Pred→"] + all_labels
    
    for true_label in all_labels:
        row = [true_label]
        for pred_label in all_labels:
            row.append(confusion_matrix[true_label][pred_label])
        matrix_table.add_row(row)
    
    print(matrix_table)

if __name__ == "__main__":
    print("🔄 Loading model and evaluating...")
    nlp = spacy.load("ner_model")
    metrics, confusion_matrix = evaluate_ner(nlp, test_data)
    print_metrics(metrics, confusion_matrix)
