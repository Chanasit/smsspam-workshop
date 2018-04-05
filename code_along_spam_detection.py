# coding: utf-8

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('spamDetection').getOrCreate()


# Read spam email data
data = spark.read.csv('smsspamcollection/SMSSpamCollection.octet-stream', inferSchema=True, sep='\t')

print('No. of rows:', data.count())
data.show(10)

data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data.show(10)

from pyspark.sql.functions import length, mean, count

data = data.withColumn('length', length(data['text']))
data.show(10)

# basic check for length of emalis
data.groupBy('class').agg(mean(data["length"]).alias('average_length'), count(data["length"]).alias('count')).show()


# Time to do some data tansformation
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer

tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf-idf')
ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')

data_tokenized = tokenizer.transform(data)
data_tokenized.show(5)

data_stop_removed = stop_remove.transform(data_tokenized)
data_stop_removed.show(5)

data_count_vec = count_vec.fit(data_stop_removed).transform(data_stop_removed)
data_count_vec.show(5)

data_idf = idf.fit(data_count_vec).transform(data_count_vec)
data_idf.show(5)

data_final = ham_spam_to_numeric.fit(data_idf).transform(data_idf)
data_final.show(5)


#That's a long process, let's use Pipeline
#We also want to do VectorAssembler
from pyspark.ml.feature import VectorAssembler
clean_up = VectorAssembler(inputCols=['tf-idf','length'], outputCol='features')

from pyspark.ml import Pipeline

data_prep_pipe = Pipeline(stages=[tokenizer, stop_remove, count_vec, idf, clean_up, ham_spam_to_numeric])

data_clean = data_prep_pipe.fit(data).transform(data)

data_clean.show(5)

# not important for running a model but to make it look cleaner in this notebook
data_clean = data_clean.select('class', 'label', 'features')
data_clean.show(5)


# Build a model and transform test_set
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()

train_set, test_set = data_clean.randomSplit([0.7, 0.3])

spam_detector = nb.fit(train_set)

test_results = spam_detector.transform(test_set)

test_results.show(5)


# Evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()

acc_eval.extractParamMap()

print('Model accuracy:')
print(acc_eval.evaluate(test_results))
