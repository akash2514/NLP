import spacy
import pickle
import random
import os
import warnings
from spacy.util import minibatch, compounding
os.chdir(r'D:\python\NLP projects\CV_resume_parsing\Resume-and-CV-Summarization-and-Parsing-with-Spacy-in-Python')

nlp = spacy.blank('en')

train_data = pickle.load(open(r'D:\python\NLP projects\CV_resume_parsing\Resume-and-CV-Summarization-and-Parsing-with-Spacy-in-Python\train_data.pkl','rb'))

def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    # only train NER
    with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            for itn in range(70):
                print("Starting iteration" + str(itn))
                random.shuffle(train_data)
                losses = {}
                index = 0
                for text, annotations in train_data:
                    #print(index)
                    try:
                        nlp.update(
                            [text],
                            [annotations],
                            drop=0.20,
                            sgd=optimizer,
                            losses = losses
                        )
                    except Exception as e:
                        pass

                print("Losses", losses)




train_model(train_data)
nlp.to_disk('resume_ner_akash')
# nlp_model = spacy.load('resume_ner_akash')
# doc = nlp_model(train_data[0][0])
#
# for ent in doc.ents:
#     print(ent.label_,"-----" ,ent.text)
'''
Name ----- Govardhana K
Designation ----- Senior Software Engineer
Location ----- Bengaluru
Email Address ----- indeed.com/r/Govardhana-K/ b2de315d95905b68
Designation ----- Senior Software Engineer
Designation ----- Senior Consultant
Companies worked at ----- Oracle
Companies worked at ----- Oracle
Designation ----- Associate Consultant
Companies worked at ----- Oracle
Degree ----- B.E in Computer Science Engineering
College Name ----- Adithya Institute of Technology
Skills ----- APEX. (Less than 1 year), Data Structures (3 years), FLEXCUBE (5 years), Oracle (5 years), Algorithms (3 years)
'''

# test run
file = open('Alice Clark CV.txt','r')
text = file.read()
text = ' '.join(text.split('\n'))
nlp_model = spacy.load('resume_ner_akash')
doc = nlp_model(text)

for ent in doc.ents:
    print(ent.label_,"-----" ,ent.text)
