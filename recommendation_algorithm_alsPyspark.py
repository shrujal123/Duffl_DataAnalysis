from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
import pandas as pd

SparkContext.setSystemProperty('spark.executor.memory', '4g')
sc = SparkContext("local", "App Name")

sqlContext = SQLContext(sc)

df = sqlContext.read.format( 'com.databricks.spark.csv' ).options( header='true', inferSchema = 'true' ).load( 'customer_product_ids_list.csv' )
new_df = df.sample(0.01)

(training, test) = new_df.randomSplit([0.8, 0.2])
als = ALS(userCol = "customer_id", itemCol = "product_id", ratingCol = "quantity", coldStartStrategy = "drop", nonnegative = True)


param_grid = ParamGridBuilder().addGrid(als.rank, [12]).addGrid(als.maxIter, [18]).addGrid(als.regParam, [.17]).build()
#rank of latent factor matrices, how many times to alternate, prevent overfitting regularization parameter
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "quantity", predictionCol = "prediction")
#cross validation
tvs = TrainValidationSplit(estimator = als, estimatorParamMaps = param_grid, evaluator = evaluator)

model = tvs.fit(training)

best_model = model.bestModel

predictions = best_model.transform(test)
rmse = evaluator.evaluate(predictions)

print("RMSE = " + str(rmse))
print(" Rank:"), best_model.rank
print(" MaxIter:"), best_model._java_obj.parent().getMaxIter()
print(" RegParam:"), best_model._java_obj.parent().getRegParam()

#predictions.sort("customer_id", "quantity").show()

user_recs = best_model.recommendForAllUsers(10)
user_recs = user_recs.withColumn("product_id", user_recs["recommendations"].product_id).drop('recommendations')

user_recs_df = user_recs.toPandas()
#user_recs_df

def recs_for_user(df, customer_id):
    recs_list = user_recs_df.loc[user_recs_df["customer_id"] == customer_id, "product_id"].iloc[0]
    return recs_list

recs_for_user(user_recs_df, #userid)
