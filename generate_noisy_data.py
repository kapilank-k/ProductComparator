import random

# Vocabulary sets
materials = ["TMT", "OPC", "PPC", "Steel", "Rebar", "Strand"]
grades = ["Fe 500D", "Fe 550", "Fe 415", "53", "43", "CLASS I", "CLASS II"]
forms = ["Loose", "Packed", "Bulk", "Straight Bars", "Coil", "Bagged"]
diameters = ["10 mm", "12 mm", "16 mm", "20 mm", "25 mm", "8.00 mm", "12.00 mm"]
lengths = ["12000 mm", "6000 mm", "9000 mm", "11000 mm", "10000 mm"]
standards = ["IS 1786", "IS 12269", "IS 456", "IS 10262", "ASTM A615"]

# Add optional noise words
material_noise = ["steel", "bar", "rod", "cement type", "grade of", "kind"]
form_noise = ["form", "packing", "delivery", "packing type"]
extra_words = ["available", "as per", "standard", "approx", "length", "diameter", "in stock", "confirmed"]

def create_sample():
    mat = random.choice(materials)
    mat_label = mat
    grade = random.choice(grades)
    grade_label = grade
    form = random.choice(forms)
    form_label = form
    dia = random.choice(diameters)
    dia_label = dia
    length = random.choice(lengths)
    len_label = length
    standard = random.choice(standards)
    std_label = standard

    # Add noise around fields
    sentence = f"{random.choice(material_noise)} {mat} {random.choice(extra_words)} {grade} {random.choice(form_noise)} {form}, dia: {dia}, length={length} conf. to {standard}"

    # Normalize text to get start/end character indices
    sentence = sentence.replace("  ", " ")

    ents = []
    start = sentence.find(mat)
    if start != -1:
        ents.append((start, start + len(mat_label), "Material"))

    start = sentence.find(grade)
    if start != -1:
        ents.append((start, start + len(grade_label), "Grade"))

    start = sentence.find(form)
    if start != -1:
        ents.append((start, start + len(form_label), "Form"))

    start = sentence.find(dia)
    if start != -1:
        ents.append((start, start + len(dia_label), "Diameter"))

    start = sentence.find(length)
    if start != -1:
        ents.append((start, start + len(len_label), "Length"))

    start = sentence.find(standard)
    if start != -1:
        ents.append((start, start + len(std_label), "Standard"))

    return (sentence, {"entities": ents})

# Generate noisy training samples
data = [create_sample() for _ in range(1000)]

# Save to noisy_training_data.py
with open("noisy_training_data.py", "w", encoding="utf-8") as f:
    f.write("TRAIN_DATA = [\n")
    for text, ann in data:
        f.write(f"    ({text!r}, {ann}),\n")
    f.write("]\n")

print("✅ Generated 1000 noisy samples to noisy_training_data.py")
