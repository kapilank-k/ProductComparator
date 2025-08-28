import random
import re
from typing import List, Tuple, Dict, Any
import sys
sys.path.append(".")
from train_split import TRAIN_DATA

# ---
# DATA AUGMENTATION SCRIPT
# Generates diverse training examples using various techniques
# ---

class DataAugmenter:
    def __init__(self):
        """Initialize the data augmenter with augmentation strategies"""
        self.materials = ["TMT", "OPC", "PC STRAND", "CEMENT", "STEEL", "HT STRAND", "REBAR"]
        self.grades = ["Fe500", "Fe500D", "Fe550", "Fe415", "OPC 43", "OPC 53", "Class I", "Class II"]
        self.diameters = ["8 mm", "10 mm", "12 mm", "12.5 mm", "16 mm", "20 mm", "25 mm"]
        self.lengths = ["6000 mm", "12000 mm", "16000 mm", "18000 mm"]
        self.forms = ["Loose", "Bulk", "Packed", "Bag", "Coil", "Bundle", "Straight bars"]
        self.standards = ["IS 1786", "IS 12269", "IS 14268", "IS 456", "IS 8112", "IS 6003"]
        
        # Synonyms and variations
        self.synonyms = {
            "TMT": ["TMT", "Thermo Mechanically Treated", "TMT Bars"],
            "OPC": ["OPC", "Ordinary Portland Cement", "Portland Cement"],
            "PC STRAND": ["PC STRAND", "Pre-stressed Concrete Strand", "Strand"],
            "mm": ["mm", "millimeter", "millimeters"],
            "IS": ["IS", "Indian Standard", "I.S."],
            "Fe": ["Fe", "FE", "fe"]
        }
        
        # Common typos and abbreviations
        self.typos = {
            "TMT": ["TMT", "TMTT", "TMT BAR", "TMT BARS"],
            "OPC": ["OPC", "OPCC", "OPC CEMENT"],
            "Fe500": ["Fe500", "FE500", "fe500", "Fe 500", "FE 500"],
            "mm": ["mm", "MM", "mm.", "MM."],
            "IS": ["IS", "I.S", "is", "Is"]
        }
    
    def get_entity_offsets(self, text: str, entity_value: str, label: str) -> Tuple[int, int, str]:
        """Get entity offsets in text"""
        start = text.find(entity_value)
        end = start + len(entity_value)
        return (start, end, label)
    
    def synonym_replacement(self, text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Replace entities with synonyms"""
        new_text = text
        new_entities = []
        offset = 0
        
        for start, end, label in entities:
            entity_text = text[start:end]
            
            # Find synonyms for this entity
            for key, synonyms in self.synonyms.items():
                if entity_text.upper() in [s.upper() for s in synonyms]:
                    # Choose a random synonym
                    new_entity = random.choice(synonyms)
                    
                    # Replace in text
                    new_text = new_text[:start + offset] + new_entity + new_text[end + offset:]
                    
                    # Update entity position
                    new_entities.append((start + offset, start + offset + len(new_entity), label))
                    offset += len(new_entity) - len(entity_text)
                    break
            else:
                # No synonym found, keep original
                new_entities.append((start + offset, end + offset, label))
        
        return new_text, new_entities
    
    def typo_injection(self, text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Inject common typos into entities"""
        new_text = text
        new_entities = []
        offset = 0
        
        for start, end, label in entities:
            entity_text = text[start:end]
            
            # Find typos for this entity
            for key, typos in self.typos.items():
                if entity_text.upper() in [t.upper() for t in typos]:
                    # Choose a random typo
                    new_entity = random.choice(typos)
                    
                    # Replace in text
                    new_text = new_text[:start + offset] + new_entity + new_text[end + offset:]
                    
                    # Update entity position
                    new_entities.append((start + offset, start + offset + len(new_entity), label))
                    offset += len(new_entity) - len(entity_text)
                    break
            else:
                # No typo found, keep original
                new_entities.append((start + offset, end + offset, label))
        
        return new_text, new_entities
    
    def case_variation(self, text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Create case variations of entities"""
        new_text = text
        new_entities = []
        offset = 0
        
        for start, end, label in entities:
            entity_text = text[start:end]
            
            # Create case variations
            variations = [
                entity_text.upper(),
                entity_text.lower(),
                entity_text.title(),
                entity_text.capitalize()
            ]
            
            # Choose a random variation
            new_entity = random.choice(variations)
            
            # Replace in text
            new_text = new_text[:start + offset] + new_entity + new_text[end + offset:]
            
            # Update entity position
            new_entities.append((start + offset, start + offset + len(new_entity), label))
            offset += len(new_entity) - len(entity_text)
        
        return new_text, new_entities
    
    def spacing_variation(self, text: str, entities: List[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Add spacing variations"""
        new_text = text
        new_entities = []
        offset = 0
        
        for start, end, label in entities:
            entity_text = text[start:end]
            
            # Add random spacing
            if random.random() < 0.3:  # 30% chance
                # Add extra spaces
                new_entity = " ".join(entity_text.split())
                if random.random() < 0.5:
                    new_entity = new_entity.replace(" ", "  ")  # Double spaces
                
                # Replace in text
                new_text = new_text[:start + offset] + new_entity + new_text[end + offset:]
                
                # Update entity position
                new_entities.append((start + offset, start + offset + len(new_entity), label))
                offset += len(new_entity) - len(entity_text)
            else:
                # Keep original
                new_entities.append((start + offset, end + offset, label))
        
        return new_text, new_entities
    
    def template_based_augmentation(self, num_examples: int = 100) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate new examples using templates with variations"""
        templates = [
            "{material} {grade} {form} {diameter} {length} {standard}",
            "{grade} {material} {form} {diameter} {standard}",
            "{material} {grade} {diameter} {length} {standard}",
            "{form} {material} {grade} {standard}",
            "{material} {form} {grade} {diameter} {standard}",
            "{material} {grade} {diameter} {standard}",
            "{grade} {material} {diameter} {length}",
            "{material} {diameter} {length} {standard}",
            "{form} {material} {grade} {diameter}",
            "{material} {grade} {form} {standard}"
        ]
        
        augmented_data = []
        
        for _ in range(num_examples):
            template = random.choice(templates)
            
            # Randomly select values
            material = random.choice(self.materials)
            grade = random.choice(self.grades)
            diameter = random.choice(self.diameters)
            length = random.choice(self.lengths)
            form = random.choice(self.forms)
            standard = random.choice(self.standards)
            
            # Generate text
            text = template.format(
                material=material,
                grade=grade,
                diameter=diameter,
                length=length,
                form=form,
                standard=standard
            )
            
            # Extract entities
            entities = []
            for val, label in [
                (material, "Material"),
                (grade, "Grade"),
                (diameter, "Diameter"),
                (length, "Length"),
                (form, "Form"),
                (standard, "Standard")
            ]:
                if val in text:
                    entities.append(self.get_entity_offsets(text, val, label))
            
            augmented_data.append((text, {"entities": entities}))
        
        return augmented_data
    
    def augment_existing_data(self, data: List[Tuple[str, Dict[str, Any]]], 
                            augmentation_factor: int = 2) -> List[Tuple[str, Dict[str, Any]]]:
        """Augment existing data using various techniques"""
        augmented_data = []
        
        for text, annotations in data:
            entities = annotations.get("entities", [])
            
            # Apply different augmentation techniques
            techniques = [
                self.synonym_replacement,
                self.typo_injection,
                self.case_variation,
                self.spacing_variation
            ]
            
            # Create multiple augmented versions
            for _ in range(augmentation_factor):
                technique = random.choice(techniques)
                new_text, new_entities = technique(text, entities)
                
                # Only add if it's different from original
                if new_text != text:
                    augmented_data.append((new_text, {"entities": new_entities}))
        
        return augmented_data
    
    def generate_noisy_data(self, num_examples: int = 50) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate intentionally noisy data for robustness"""
        noisy_data = []
        
        for _ in range(num_examples):
            # Create a base template
            base_text = f"{random.choice(self.materials)} {random.choice(self.grades)} {random.choice(self.diameters)}"
            
            # Add noise
            noise_types = [
                lambda t: t + f" {random.choice(['extra', 'additional', 'supplementary'])} info",
                lambda t: t.replace("mm", random.choice(["mm", "MM", "millimeter", "millimeters"])),
                lambda t: t + f" {random.choice(['quality', 'premium', 'standard'])} grade",
                lambda t: t.replace("IS", random.choice(["IS", "I.S", "Indian Standard"])),
                lambda t: t + f" {random.choice(['packaging', 'delivery', 'storage'])}: {random.choice(self.forms)}"
            ]
            
            # Apply random noise
            for _ in range(random.randint(1, 3)):
                noise_func = random.choice(noise_types)
                base_text = noise_func(base_text)
            
            # Extract entities (simplified)
            entities = []
            for material in self.materials:
                if material in base_text:
                    entities.append(self.get_entity_offsets(base_text, material, "Material"))
                    break
            
            for grade in self.grades:
                if grade in base_text:
                    entities.append(self.get_entity_offsets(base_text, grade, "Grade"))
                    break
            
            # Add diameter if present
            diameter_match = re.search(r'(\d+\.?\d*\s*mm)', base_text, re.IGNORECASE)
            if diameter_match:
                entities.append(self.get_entity_offsets(base_text, diameter_match.group(1), "Diameter"))
            
            noisy_data.append((base_text, {"entities": entities}))
        
        return noisy_data

def main():
    """Main function to generate augmented data"""
    print("🚀 Starting data augmentation...")
    
    augmenter = DataAugmenter()
    
    # 1. Augment existing data
    print("📊 Augmenting existing training data...")
    augmented_existing = augmenter.augment_existing_data(TRAIN_DATA, augmentation_factor=2)
    print(f"✅ Generated {len(augmented_existing)} augmented examples from existing data")
    
    # 2. Generate template-based examples
    print("📝 Generating template-based examples...")
    template_examples = augmenter.template_based_augmentation(num_examples=200)
    print(f"✅ Generated {len(template_examples)} template-based examples")
    
    # 3. Generate noisy data
    print("🔊 Generating noisy data for robustness...")
    noisy_examples = augmenter.generate_noisy_data(num_examples=100)
    print(f"✅ Generated {len(noisy_examples)} noisy examples")
    
    # Combine all data
    all_augmented = TRAIN_DATA + augmented_existing + template_examples + noisy_examples
    
    # Save augmented data
    with open("augmented_training_data_enhanced.py", "w", encoding="utf-8") as f:
        f.write("# Enhanced Augmented Training Data\n")
        f.write("# Generated using multiple augmentation techniques\n\n")
        f.write("TRAIN_DATA = [\n")
        for text, ann in all_augmented:
            f.write(f"    ({text!r}, {ann}),\n")
        f.write("]\n")
    
    print(f"\n🎉 Total augmented data: {len(all_augmented)} examples")
    print(f"📁 Saved to 'augmented_training_data_enhanced.py'")
    
    # Print statistics
    entity_counts = {}
    for _, ann in all_augmented:
        for start, end, label in ann.get("entities", []):
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print(f"\n📈 Entity distribution:")
    for entity, count in sorted(entity_counts.items()):
        print(f"  {entity}: {count}")

if __name__ == "__main__":
    main() 