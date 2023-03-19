import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger') #for pos
nltk.download('wordnet') #for lemmatizing
nltk.download('maxent_ne_chunker') #for named entity recognition
nltk.download("words") # for named entity recognition


class NLPProject:
    def __init__(self, text):
        print('################### Processing text ########################')
        self.text = text
        self.sentences = nltk.sent_tokenize(self.text)
        print('Text in a list :',self.sentences)
        self.stop_words = set(stopwords.words('english'))
        print('Stop word in the text : ',self.stop_words)
        self.stemmer = PorterStemmer()
        print('###########################################')

    def tokenize_words(self, sentence):
        words = nltk.word_tokenize(sentence)
        return words

    def remove_stop_words(self, words):
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return filtered_words

    def perform_stemming(self, words):
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in words]
        return stemmed_words
    def perform_pos_tagging(self, stemmed_words):
        pos_tagging = nltk.pos_tag(stemmed_words)
        return pos_tagging
    def perform_lemmatizing(self,stemmed_words):
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(stemmed_word) for stemmed_word in stemmed_words]
        return lemmatized_words

    def extract_ne(self,pos_tagging):
        tree = nltk.ne_chunk(pos_tagging, binary=True)
        return set(" ".join(i[0] for i in t)
        for t in tree
                 if hasattr(t, "label") and t.label() == "NE")
    def process_text(self):
        processed_sentences = []
        tokenized = []
        remStop = []
        stemWrd = []
        ptag = []

        print(self)
        for sentence in self.sentences:
            words = self.tokenize_words(sentence)
            tokenized.append(words)
            words = self.remove_stop_words(words)
            remStop.append(words)
            stemmed_words = self.perform_stemming(words)
            stemWrd.append(stemmed_words)
            pos_tagging = self.perform_pos_tagging(stemmed_words)
            ptag.append(pos_tagging)
            lemmatizing = self.perform_lemmatizing(stemmed_words)
            processed_sentences.append(lemmatizing)
        print('Tokenized words : ',tokenized)
        print('After removing stop words : ',remStop)
        print('After performing stemming : ',stemWrd)
        print('After performing pos : ',ptag)
        print('After performing lemmatizing : ',processed_sentences)
        return processed_sentences



def main():

    input_text = "We implemented this setup to capture dynamic biological processes in challenging environments " \
                 "using the opti-cal arrangement and resolution obtained above. To this end, we first optimized the " \
                 "time-lapse imaging of plant samples using the camera’s built-in intervalometer to establish an " \
                 "adequate frequency for visualizing dynamic processes. This setup allowed us to operate the camera " \
                 "autonomously and use a computer to control the system for more specific and intricate tasks, " \
                 "such as automated focus stacking operations."
    # Create an instance of the NLPProject class
    project = NLPProject(input_text)
    # Call the process_text() method to process the input text
    processed_text = project.process_text()

    # Print the processed text
    print('#################Processed sentences ######################')
    for sentence in processed_text:
        print(sentence)


if __name__ == "__main__":
    main()
