import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any

def stratified_split(data: List[Tuple[str, Dict[str, Any]]], train_ratio=0.7, dev_ratio=0.15):
    """
    Split data while maintaining entity distribution across splits
    """
    # Group examples by entity combinations
    entity_groups = defaultdict(list)
    for example in data:
        text, annots = example
        # Create a key based on entity types present
        entity_types = tuple(sorted(set(ent[2] for ent in annots["entities"])))
        entity_groups[entity_types].append(example)
    
    train_data = []
    dev_data = []
    test_data = []
    
    # Split each group proportionally
    for group in entity_groups.values():
        random.shuffle(group)
        n = len(group)
        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))
        
        train_data.extend(group[:train_end])
        dev_data.extend(group[train_end:dev_end])
        test_data.extend(group[dev_end:])
    
    return train_data, dev_data, test_data

if __name__ == "__main__":
    # Load all available training data
    from base_training_data import TRAIN_DATA as base_data
    from augmented_training_data import TRAIN_DATA as augmented_data
    from noisy_training_data import TRAIN_DATA as noisy_data
    
    # Combine all data
    all_data = base_data + augmented_data + noisy_data
    
    # Perform stratified split
    train_data, dev_data, test_data = stratified_split(all_data)
    
    # Save splits to separate files
    def save_data(data, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("TRAIN_DATA = [\n")
            for text, ann in data:
                f.write(f"    ({text!r}, {ann}),\n")
            f.write("]\n")
    
    save_data(train_data, "train_split.py")
    save_data(dev_data, "dev_split.py")
    save_data(test_data, "test_split.py")
    
    # Print statistics
    print("\n📊 Data Split Statistics:")
    print(f"Total examples: {len(all_data)}")
    print(f"Train set: {len(train_data)} examples ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"Dev set:   {len(dev_data)} examples ({len(dev_data)/len(all_data)*100:.1f}%)")
    print(f"Test set:  {len(test_data)} examples ({len(test_data)/len(all_data)*100:.1f}%)")
