from __future__ import print_function
# $example on$
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession
import sys

#ml is for dataframe, mllib is for RDD
#sparksession is for dataframe, sparkcontext if for RDD
if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("AdultCensus")\
        .getOrCreate()

    #load data
    dataset = spark.read.format("csv").load("/logistic_regression/input/census_clean.csv", 
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
    features = assembler.transform(data_all)
    #features.show()

    #Split data to train and test set
    train, test = features.randomSplit((0.8, 0.2), seed=0)

    #Logistic Regression Model
    logistic = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 10)

    # fitting the model
    model = logistic.fit(train)

    #print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    #predict test set
    prediction = model.transform(test)
    #prediction.show()

    # get testing accuracy
    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(prediction)
    print(accuracy)




