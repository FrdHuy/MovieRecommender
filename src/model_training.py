from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def train_model(ratings_df):
    (training_data, test_data) = ratings_df.randomSplit([0.8, 0.2])

    # Initialize ALS model
    als = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )

    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 50, 100]) \
        .addGrid(als.maxIter, [5, 10, 15]) \
        .addGrid(als.regParam, [0.05, 0.1, 0.5]) \
        .build()

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    cross_validator = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5
    )
    cv_model = cross_validator.fit(training_data)
    best_model = cv_model.bestModel

    return best_model, test_data

def evaluate_model(model, test_data):
    predictions = model.transform(test_data)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Square Error (RMSE) = {rmse:.4f}")
    return rmse
