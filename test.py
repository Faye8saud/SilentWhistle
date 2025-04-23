import spacy

nlp = spacy.load("ner_arabic_model")

examples = [
    "الكرة لمست يد اللاعب",     # from training
    "ركنية لصالح الفريق",        # from training
    "الريمونتادا بدأت",         # from training
    "الفريق يضغط ضغط عالي جدا",  # from training
    "محمد سجل هدف",              # not from training
    "هجمة مرتدة قوية",           # variation
]

for sentence in examples:
    doc = nlp(sentence)
    print(f"\nSentence: {sentence}")
    for ent in doc.ents:
        print(f" → Entity: '{ent.text}' | Label: {ent.label_}")
