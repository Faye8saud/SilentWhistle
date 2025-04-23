import spacy
from spacy.training.example import Example
from ner_training_data import TRAIN_DATA

nlp = spacy.blank("ar")  # Arabic pipeline
ner = nlp.add_pipe("ner")


for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Training
nlp.begin_training()
for itn in range(30):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example])

nlp.to_disk("ner_arabic_model")
print("Custom Arabic NER model saved!")
