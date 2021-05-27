from __future__ import print_function
# $example on$
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession
import sys

#ml is for dataframe, mllib is for RDD
#sparksession is for dataframe, sparkcontext if for RDD
if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("AdultCensusDecisionTree")\
        .getOrCreate()

    #load data
    dataset = spark.read.format("csv").load("/decision_tree_classification/input/census_clean.csv",
                                            header="true", inferSchema="true",
                                            ignoreLeadingWhiteSpace='true',
                                            ignoreTrailingWhiteSpace='true')
    #remove whitespace, replace feature name with a '.' to '' and convert to a dataframe
    data_list = []
    for col in dataset.columns:
        new_name = col.strip()
        new_name = "".join(new_name.split())
        new_name = new_name.replace('.','')
        data_list.append(new_name)
    print(data_list)
    data = dataset.toDF(*data_list)
    #data.show()

    #add a label column using feature income
    index = StringIndexer(inputCol = 'income', outputCol = 'label')
    data_all =index.fit(data).transform(data)

    #vectorize features
    assembler = VectorAssembler(inputCols= ['age', 'workclass', 'maritalstatus',
                                            'educationnum', 'occupation', 'relationship',
                                            'race', 'sex', 'capitalgain', 'capitalloss',
                                            'hoursperweek', 'nativecountry'],
                                            outputCol="features")
    feature = assembler.transform(data_all)
    #feature.show()

    #Split data to train and test set
    train, test = feature.randomSplit((0.8, 0.2), seed=0)

    # Train a DecisionTree model.
    decision_tree = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    model = decision_tree.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select (prediction, true label) and compute test error, accuracy 
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Testing accuracy: ", accuracy)
    print("Test Error = %g " % (1.0 - accuracy))

