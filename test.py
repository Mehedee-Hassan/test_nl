# import nltk.data
# import pickle
# tagger = pickle.load(open("D:\corpus\corpus_aubt.pickle",encoding='UTF-8'))
# tagger.tag(['আমি','ভাত','খাই'])

from nltk.corpus.reader import TaggedCorpusReader
import nltk
from nltk.corpus import brown


reader = TaggedCorpusReader('D:\\corpus', r'.*\.pos',sep='_',encoding='utf-8')


print('words =',reader.words())
print('tagged words =',reader.tagged_words())
print('tagged sentence =',reader.tagged_sents())
print('sents =',reader.sents()[0])

unigram_tagger = nltk.UnigramTagger(reader.tagged_sents())
print('evaluate =',unigram_tagger.evaluate(reader.tagged_sents()))

f1 = open("in.txt", "r",encoding='utf-8')

print(unigram_tagger.tag(reader.tagged_sents()[5]));


f2 = open("out2.txt", "a",encoding='utf-8')
f2.write("\n")
# f.write(str)

# var = unigram_tagger.tag(reader.tagged_sents()[5])

# teststring = str('বলা চলে পবিত্র মোষের দুধ দোয়া ,', 'utf-8')
# teststring = ['মোদের', 'আয়তন', 'খুব', 'বড়ো' ,'নয়', '৷']
teststring = ['আমি', 'ভাত', 'খাই', '৷']

print(reader.sents()[0])
print(teststring)

var = unigram_tagger.tag(teststring)

for v in var:
    print (v)
    f2.write(v[0])
    f2.write(' ')
    if v[1]:
        f2.write('\\')
        f2.write(v[1])
    else:
        f2.write('\\')
        f2.write('NONE')
    f2.write('  ')

# for a in reader.tagged_words():
#     f.write(" ")
#     f.write(a[0])

    


