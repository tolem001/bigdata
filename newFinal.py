from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("MLlib StudentMentalHealthML Prediction").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'StudentMentalHealthML' into a Spark DataFrame
smh_df = spark.sql("SELECT gender, course, current_year, gpa, marital_status, depression, anxiety, panic_attack, treatment, age FROM Student_Mental_Health")

# Step 3: Handle null values by either dropping or filling them
smh_df = smh_df.na.drop()  # Drop rows with null values

# Step 4: Index categorical columns
categorical_cols = ["gender", "course", "current_year", "gpa",
                    "marital_status", "depression", "anxiety",
                    "panic_attack", "treatment"]

for col in categorical_cols:
    indexer = StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip")
    smh_df = indexer.fit(smh_df).transform(smh_df)

# Step 5: Assemble features into a vector
feature_cols = [col+"_index" for col in categorical_cols]
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)
assembled_df = assembler.transform(smh_df).select("features", "age")

# Step 6: Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 7: Initialize and train a Linear Regression model
lr = LinearRegression(labelCol="age")
lr_model = lr.fit(train_data)

# Step 8: Evaluate the model on the test data
test_results = lr_model.evaluate(test_data)

# Step 9: Print the model performance metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase with happybase (using the provided pattern) ----
# Example data (row_key, column_family:column, value) populated with the metrics
data = [
    ('metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'cf:r2',   str(test_results.r2)),
]

# Function to write data to HBase inside each partition
def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('my_table')  # Update table name
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

# Parallelize data and apply the function with foreachPartition
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 10: Stop the Spark session
spark.stop()