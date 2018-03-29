import re
import string
import nltk
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

class TextPreprocessing():
	"""Provides ways of preprocessing text data."""
	
	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.punctuation = list(string.punctuation)
		self.stop_words = stopwords.words('english') + self.punctuation + ['\\n'] + ['quot']

	def tokenize_document(self, text):
		"""Preprocess a whole raw document.

		Args:
			text (str): Raw string of text.

		Return:
			Nested lists where each list is a tokenized sentence of the document. Example: [['foo', 'bar', 'hello', 'world'], ['one', 'two', 'three'], ... ].

		"""
		return [self.tokenize_sentence(sentence) for sentence in self.text2sentences(text)]
	
	def tokenize_sentence(self, text):
		"""Preprocess a raw string/sentence of text.

		Args:
			text (str): Raw string of text.

		Return:
			tokens (list, str): Preprocessed tokens.

		"""
		tokens = self.create_tokens(text)
		# pos_tagged_tokens = self.get_token_pos_tag(tokens)
		# tokens_with_ner = self.get_continuous_chunks(pos_tagged_tokens)
		# lemmatized_tokens = self.lemmatize_tokens(pos_tagged_tokens)
		tokens = self.remove_stop_words(tokens)      
		return tokens

	def text2sentences(self, text):
		"""Split raw text to sentences.

		Args:
			text (str): Raw text data.

		Return:
			List of strings, where each element is a sentence.

		"""
		return nltk.sent_tokenize(text)
	
	def create_tokens(self, text):
		"""Split a string into tokens.

		Args:
			text (str): Raw text data

		Return:
			tokens (str): The pieces of the string.

		"""
		regex_str = ["http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),](?:%[0-9a-f][0-9a-f]))+", 
					 "(?:\w+-\w+){2}",
					 "(?:\w+-\w+)", 
					 "(?:\\\+n+)", 
					 "(?:@[\w_]+)", 
					 "<[^>]+>", 
					 "(?:\w+'\w)", 
					 "(?:[\w_]+)", 
					 "(?:\S)"]

		# Create the tokenizer which will be case insensitive and will ignore space.
		tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

		# Find all patterns and tokenize them
		tokens = tokens_re.findall(text)
		return tokens
	
	def get_token_pos_tag(self, tokens):
		"""Find the part of speech tag of the tokens of a sentence using the Penn Treebank.

		Args:
			tokens (list, str): Collection of the entities that make up a sentence.

		Return:
			A list of tuples. Example: [tuple('foo', POS-tag), tuple('bar', POS-tag) ... ]

		"""
		return nltk.pos_tag(tokens)
	
	def get_continuous_chunks(self, tagged_tokens):
		"""Find name entities in a sentence.

		Args:
			tagged_ner_tokens (list, tuple): Tokens with their part of speech tag. Example: [tuple('foo', POS-tag), tuple('bar', POS-tag) ... ]

		Return:
			ner_tokens (list, tuple): Tokens with their part of speech tag and with NER done.
		
		"""
		chunks = nltk.ne_chunk(tagged_tokens)
		ner_tokens = []
		for chunk in chunks:
			current_chunk = []
			if type(chunk) == Tree:
				current_chunk.append("_".join([token for token, pos in chunk.leaves()]))
				ner_tokens.extend(current_chunk)
			else:
				ner_tokens.append(chunk)
				continue

		return ner_tokens

	def lemmatize_tokens(self, tagged_tokens):
		"""Lemmatize tokens based on their POS-tag. Keep unchanged the Name Entities. Lowercase them as well.

		Args:
			tagged_tokens (list, tuple): Tokens with their part of speech tag. Example: [tuple('foo', POS-tag), tuple('bar', POS-tag) ... ]

		Return:
			A list of lemmatized tokens.

		"""
		return [self.lemmatizer.lemmatize(tagged_token[0].lower(), self.get_wordnet_pos(tagged_token[1])) 
				if type(tagged_token) == tuple else tagged_token.lower() for tagged_token in tagged_tokens]

	def get_wordnet_pos(self, treebank_tag):
		"""Map the treebank tags to WordNet part of speech names. This is used for the lemmatization.

		Args:
			treebank_tag (str): The POS-tag of a token.

		Return:
			WordNet's names for these tags.

		"""
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		elif treebank_tag.startswith('S'):
			return wordnet.ADJ_SAT
		else:
			return wordnet.NOUN

	def remove_stop_words(self, tokens):
		"""Remove the stop words and punctuation of a tokenized sentence.

		Args:
			tokens (list, str): Collection of the entities that make up a sentence.

		Return:
			A filtered list of tokens.

		"""
		filtered_tokens = [token.lower() for token in tokens 
							if token not in self.stop_words 
							and '<' not in token
							and len(token) > 2
							and any(x in token for x in ('1234567890')) == False
							and any(x in token for x in ('qwertyuioplkjhgfdaszxcvbnm-'))]

		return filtered_tokens

	def text_aggregator(self, sentences):
		"""Join sentences using a '.'"""
		return '. '.join(sentences)

	def delete_short_sents(self, sentences):
		"""Delete sentences that have less than two tokens."""
		return [sentence for sentence in sentences if len(self.create_tokens(sentence)) > 2]

	def fix_underscores(tokens):
		"""Replace '-' with an underscore."""
		return [token.replace('-', '_') for token in tokens]

	def appends_bigrams(tokens, min_freq=2, top_n=10):
		"""Find two-word collocations, based on how often they occur together and append them to the document.

		Args:
			tokens (str): Tokens to examine for bigram collocations.
			min_freq (int): Minimum number of times that two words have to collocate in order to be counted as a bigram.
			top_n (int): The number of the most frequent bigrams to be considered.
		
		Return:
			tokens (list, str): Tokens with bigram collocation appended at the end of the list.
		
		"""
		bigram_measures = nltk.collocations.BigramAssocMeasures()
		finder = BigramCollocationFinder.from_words(tokens)
		finder.apply_freq_filter(min_freq)
		tokens.extend(['_'.join(token) for token in finder.nbest(bigram_measures.pmi, top_n)])
		return tokens

class ProcessWebsiteText(TextPreprocessing):
	def __init__(self):
		super().__init__()
		
	def tokenize_website(self, text_chunks):
		"""Preprocess a website's text, as it was scraped from the MegaScraper.

		Args:
			text_chunks (list, str): Website's text, where the elements were created by splitting the HTML page to the \n.

		Return:
			tokens (list, str): Preprocessed tokens.

		"""
		tokens = self.tokenize_document(self.text_aggregator(self.delete_short_sents(text_chunks)))
		return tokens

