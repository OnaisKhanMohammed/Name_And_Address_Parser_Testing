import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
 
# Step Two: Load Data
 
sentence = "Dr. John Talburt, JR , 1701 Darren Cirr E, 10998"

# Step Three: Tokenise, find parts of speech and chunk words 

for sent in nltk.sent_tokenize(sentence):
  for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
     if hasattr(chunk, 'label'):
        print(chunk.label(), ' '.join(c[0] for c in chunk))