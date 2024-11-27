from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType
from pyspark.sql.functions import from_unixtime, col

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("MovieRecommender") \
    .getOrCreate()

def load_and_preprocess_data():
    ratings_path = "data/ml-100k/u.data"
    movies_path = "data/ml-100k/u.item"

    ratings_schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("rating", IntegerType(), True),
        StructField("timestamp", IntegerType(), True),
    ])

    ratings_df = spark.read.csv(
        ratings_path,
        sep='\t',
        schema=ratings_schema,
        header=False
    )

    ratings_df = ratings_df.withColumn("timestamp", from_unixtime(col("timestamp")).cast(TimestampType()))

    movies_schema = StructType([
        StructField("movie_id", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("release_date", StringType(), True),
        StructField("video_release_date", StringType(), True),
        StructField("IMDb_URL", StringType(), True),
    ])

    movies_df = spark.read.csv(
        movies_path,
        sep='|',
        schema=movies_schema,
        header=False,
        encoding='ISO-8859-1'
    ).select("movie_id", "title")

    return ratings_df, movies_df
