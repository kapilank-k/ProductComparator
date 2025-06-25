import spacy
from spacy.tokens import DocBin
from base_training_data import TRAIN_DATA as TRAIN
from test_data import TEST_DATA as DEV

nlp = spacy.blank("en")

def convert(data, output_file):
    db = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_file)
    print(f"✅ Saved {len(data)} records to {output_file}")

convert(TRAIN, "train.spacy")
convert(DEV, "dev.spacy")
