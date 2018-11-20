import itertools
from nltk.tbl import Template
from nltk.tag import brill, brill_trainer
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import brown
import nltk


# dataset making part
from nltk.corpus.reader import TaggedCorpusReader
import nltk
from nltk.corpus import brown


reader = TaggedCorpusReader('F:\\programming\\nltk\\tagger\\test_nl\\data\\corpus\\t1', r'.*\.pos',sep='/',encoding='utf-8')


file_test = open("F:/programming/nltk/test_nl-master/data/corpus/t1/corpus_custom1_low.pos","r",encoding='utf-8') 

corpusArray = []
# with open("F:\\programming\\nltk\\tagger\\test_nl\\data\\corpus\\t1\\corpus_custom1_low.pos","r",encoding='utf-8') as f:


lines = file_test.readlines()

print("=*",lines[4])

for line in lines:
    # print(line) 
    words = line.split(' ')
    
    linearray = []
    for word in words:
        temp = word.split('/')
        tup = None

        if temp:
            if len(temp) > 1:
                nametemp = temp[1].strip()
                # print(nametemp)
                # if nametemp[0] == "'" and nametemp[len(nametemp)-1] == "'":
                #     nametemp = nametemp[1:len(nametemp)-1]
                
                tup = (temp[0].strip() ,nametemp)
            linearray.append(tup)

    corpusArray.append(linearray)

print("==",corpusArray[0])
print("==",corpusArray[1000])



dataset = []
for sent in reader.tagged_sents():
    temparray = []
    for tup in sent:
        if tup[0]:
            tup[0].strip()
            if tup[0] == '।' or tup[0] == ',' or tup[0] == ':' or tup[0] == '?' :
                tup = (tup[0] , 'PUNC')
        temparray.append(tup)
    dataset.append(temparray)
    

print ("\ndataset = ",dataset[:2])



# brill tagger part

def backoff_tagger(train_sents, tagger_classes, backoff=None):
	for cls in tagger_classes:
		backoff = cls(train_sents, backoff=backoff)
	
	return backoff

def word_tag_model(words, tagged_words, limit=200):
	fd = FreqDist(words)
	cfd = ConditionalFreqDist(tagged_words)
	most_freq = (word for word, count in fd.most_common(limit))
	return dict((word, cfd[word].max()) for word in most_freq)

patterns = [
	(r'^\d+$', 'CD'),
	(r'.*ing$', 'VBG'), # gerunds, i.e. wondering
	(r'.*ment$', 'NN'), # i.e. wonderment
	(r'.*ful$', 'JJ') # i.e. wonderful
]

def train_brill_tagger(initial_tagger, train_sents, **kwargs):
	templates = [
		brill.Template(brill.Pos([-1])),
		brill.Template(brill.Pos([1])),
		brill.Template(brill.Pos([-2])),
		brill.Template(brill.Pos([2])),
		brill.Template(brill.Pos([-2, -1])),
		brill.Template(brill.Pos([1, 2])),
		brill.Template(brill.Pos([-3, -2, -1])),
		brill.Template(brill.Pos([1, 2, 3])),
		brill.Template(brill.Pos([-1]), brill.Pos([1])),
		brill.Template(brill.Word([-1])),
		brill.Template(brill.Word([1])),
		brill.Template(brill.Word([-2])),
		brill.Template(brill.Word([2])),
		brill.Template(brill.Word([-2, -1])),
		brill.Template(brill.Word([1, 2])),
		brill.Template(brill.Word([-3, -2, -1])),
		brill.Template(brill.Word([1, 2, 3])),
		brill.Template(brill.Word([-1]), brill.Word([1])),
	]
	
	trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True)
	return trainer.train(train_sents, **kwargs)

def unigram_feature_detector(tokens, index, history):
	return {'word': tokens[index]}


# train brill tagger using 
# tagger
# changed brwon corpus
brown_tagged_sents = dataset #brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]




print (train_sents)

test_sents = brown_tagged_sents[size:]





t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)


default_tagger = nltk.DefaultTagger('UNK')
initial_tagger = backoff_tagger(train_sents, [nltk.UnigramTagger,nltk.BigramTagger, nltk.TrigramTagger], backoff=default_tagger)

print (initial_tagger.evaluate(test_sents))


brill_tagger = train_brill_tagger(initial_tagger, train_sents)
print (brill_tagger.evaluate(test_sents))
print("\n===========")

# print (list(brill_tagger.rules()))
print("\n===========")
teststring = 'ঝুঁকির মধ্যে দেশ : ১৯৯০ এর দশকে ও বাংলাদেশে ৫৭ শতাংশ মানুষ দারিদ্র্যসীমার নিচে বাস করত ।' 
# teststring = 'বলা চলে পবিত্র ,' 
teststring = teststring.split(' ')
# teststring = ['উন্নয়নের' ,'বিস্ময়' ,'বাংলাদেশ' , ',']
# teststring = ['বলা', 'চলে', 'পবিত্র','মোষের', 'দুধ', 'দোয়া', ',']

print(brill_tagger.tag(teststring))