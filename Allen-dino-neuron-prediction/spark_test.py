from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import numpy as np

# Initialize a Spark session
spark = SparkSession.builder.appName("ParallelLinearRegression").getOrCreate()

# Load the input images data (assuming it's stored in a cloud storage location)
#images_data = np.load("gs://your-bucket/images.npy")  # Replace with your cloud storage path
images_data=np.load('/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_movie_one.npy')

# Create a function to fit linear regression for a single file
def fit_linear_regression(file_index):
    # Load the neuron activity data from a cloud storage location
    file_name = f"gs://your-bucket/my_file_{file_index}.npy"  # Replace with your cloud storage path
    neuron_data = np.load(file_name)  # Assuming neuron_data is a NumPy array
    # Assuming you have labels for training (the ground truth you want to predict)
    # You can load these labels from cloud storage as well

    # Prepare the feature vector for linear regression
    assembler = VectorAssembler(inputCols=["image_features"], outputCol="features")
    # Assuming you have a way to associate images with neuron_data
    vectorized_data = assembler.transform(neuron_data)

    # Initialize and fit a linear regression model
    lr = LinearRegression(labelCol="activity", featuresCol="features")
    model = lr.fit(vectorized_data)

    # You can save the model to cloud storage or perform other actions as needed
    model.save(f"gs://your-bucket/models/model_{file_index}")

# Create a list of file indices (0 to 99)
file_indices = range(100)

# Use Spark to parallelize the linear regression fitting across files
spark.sparkContext.parallelize(file_indices).foreach(fit_linear_regression)

# Stop the Spark session
spark.stop()
