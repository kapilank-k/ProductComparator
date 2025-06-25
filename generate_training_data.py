import random

materials = ["TMT", "OPC", "PC STRAND", "CEMENT", "STEEL", "HT STRAND"]
grades = ["Fe500", "Fe500D", "Fe550", "Fe415", "OPC 43", "OPC 53", "Class I", "Class II"]
diameters = ["8 mm", "10 mm", "12 mm", "12.5 mm", "16 mm", "20 mm"]
lengths = ["6000 mm", "12000 mm", "16000 mm"]
forms = ["Loose", "Bulk", "Packed"]
standards = ["IS 1786", "IS 12269", "IS 14268", "IS 456", "IS 8112", "IS 6003"]

TEMPLATES = [
    "{material} {grade} {form} {diameter} {length} {standard}",
    "{grade} {material} {form} {diameter} {standard}",
    "{material} {grade} {diameter} {length} {standard}",
    "{form} {material} {grade} {standard}",
    "{material} {form} {grade} {diameter} {standard}",
]

def get_entity_offsets(text, entity_value, label):
    start = text.find(entity_value)
    end = start + len(entity_value)
    return (start, end, label)

def generate_sample():
    template = random.choice(TEMPLATES)
    material = random.choice(materials)
    grade = random.choice(grades)
    diameter = random.choice(diameters)
    length = random.choice(lengths)
    form = random.choice(forms)
    standard = random.choice(standards)

    text = template.format(
        material=material,
        grade=grade,
        diameter=diameter,
        length=length,
        form=form,
        standard=standard
    )

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
            entities.append(get_entity_offsets(text, val, label))

    return (text, {"entities": entities})

# Generate and save samples
if __name__ == "__main__":
    generated_data = [generate_sample() for _ in range(1000)]
    
    with open("augmented_training_data.py", "w", encoding="utf-8") as f:
        f.write("TRAIN_DATA = [\n")
        for text, ann in generated_data:
            f.write(f"    ({text!r}, {ann}),\n")
        f.write("]\n")
    print("✅ Generated 1000 training samples in 'augmented_training_data.py'")
