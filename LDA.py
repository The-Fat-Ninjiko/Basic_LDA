import gensim
from gensim import corpora

documents = [['apple', 'banana', 'fruits'], ['bought', 'bicycle', 'recently', 'less', 'two', 'years', 'buy', 'bike'],['colour', 'apple', 'bicycle', 'red']]

mapping = corpora.Dictionary(documents)

data = [mapping.doc2bow(word) for word in documents]

# Train LDA model
ldamodel = gensim.models.ldamodel.LdaModel(data, num_topics=2, id2word=mapping, passes=15)

# Show topics
topics = ldamodel.show_topics()
print(topics)
[(0, '0.167*"apple" + 0.154*"banana" + 0.154*"fruits" + 0.054*"colour" + 0.054*"red" + 0.053*"bicycle" + 0.052*"less" + 0.052*"bought" + 0.052*"recently" + 0.052*"years"'), (1, '0.136*"bicycle" + 0.082*"buy" + 0.082*"bike" + 0.082*"two" + 0.082*"years" + 0.082*"recently" + 0.082*"bought" + 0.082*"less" + 0.081*"red" + 0.081*"colour"')]

# Distribution of topics for the first document 
print(ldamodel.get_document_topics(data[0])) 
[(0, 0.8676003), (1, 0.13239971)]
