package com.sparkProject
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

/**
  * Created by skakeuh on 28/10/16.
  */
object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .master("local")
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /********************************************************************************
      *
      *        TP SPARK LOGISTIC REGRESSION
      *
      *
      ********************************************************************************/

    //=======================1- Mettre les données sous une forme utilisable par Spark.ML.===========

    // a------------------ load data------------------------

    val df = spark.read.parquet("/home/skakeuh/Desktop/SPARK/Projet_logistic_regression_scala/cleanedDataFrame.parquet")
    /*df.show()
    val columns = df.columns.slice(10, 20) // df.columns returns an Array. In scala arrays have a method “slice”
                                           //returning a slice of the array
    df.select(columns.map(col): _*).show(50) //

    println("\n*******number of columns", df.columns.length)
    println("\n***************number of rows", df.count)*/

    // ----------------- Mise en forme des colonnes------------------------

    val df2_labels = df.select("rowid", "koi_disposition")
    val df2_features =  df.drop("koi_disposition")
    /* df2_features.printSchema()
    / df.printSchema()*/

    // ----------------create a vector assembler----------------

    val assembler = new VectorAssembler()
      .setInputCols(df2_features.columns)
      .setOutputCol("features")

    val df_filtred = assembler.transform(df)
    // println(df_filtred.select("vector_features", "rowid").first())


    // b------------------ recentring and rescaling------------------------

    // plus besoin de rescale and recentring car la fonction StandardScaler() traite pour
    // l'instant uniquement les RDD.

    /*val scaler = new StandardScaler()
      .setInputCol("vector_features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(df_filtred)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(df_filtred)
    scaledData.show()*/

    // c------------------ Travailler avec les chaînes de caractères ------------------------

    //colonne de Strings (“CONFIRMED” ou “FALSE-POSITIVE”) transformé en '0' ou '1'
    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")

    val label_indexed = indexer.fit(df_filtred).transform(df_filtred)
    label_indexed.show()
    /*val df3_features = scaledData.join(label_indexed)
    df3_features.show(5)*/

    //=======================2- Machine Learning ===============================================

    //-------------- a- Splitter les données en Training Set et Test Set-----------------------

    // Split the data into training (90%) and test sets (10% held out for testing).
    //http://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression

    val Array(trainingData, testData) = label_indexed.randomSplit(Array(0.9, 0.1), seed = 12345)

    //-------------- b-  Entraînement du classifieur et réglage des hyper-paramètres de l’algorithme.------------------

    val lr = new LogisticRegression()
      .setElasticNetParam(1.0)  // L1-norm regularization : LASSO
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStandardization(true)  // we already scaled the data
      .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
      .setTol(1.0e-5)  // stop criterion of the algorithm based on its convergence
      .setMaxIter(300)  // a security stop criterion to avoid infinite loops

    //splitting the above 90% training into new 70% training data and new 30% validation_set for cross validation
    val Array(trainingData_model, validation_set) = trainingData.randomSplit(Array(0.7, 0.3), seed = 12345)

    // Fitting the model for first fitting before tuning
    /* val lrModel = lr.fit(trainingData)
     Print the coefficients and intercept for logistic regression
     println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")*/

    val pipeline = new Pipeline()
      .setStages(Array(lr))

    //creating the array of parameter in logarithmic scale [10e-6;;;;1] with step 0.5
    val array = -6.0 to (0.0,0.5) toArray
    val arrayLog = array.map(x => math.pow(10,x))

    //building the grid parameter for model training / testing
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, arrayLog)
      .build()

    //performing a cross validation base on grid parameter in order to have the best hyperparameter
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    // fit and train model with cross validation
    val cvModel = cv.fit(trainingData_model)

    // Make predictions on model from cross validation
    val predictions_cv = cvModel.transform(validation_set)
    val predictions = cvModel.transform(testData)

    // Select some predictives label rows to display.
    predictions_cv.select( "label", "features").show(5)
    predictions.select( "label", "features").show(5)

    // Select (prediction, true label) and compute accuracy.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy") //setting the metric for model scoring / evaluation

    //calculating the model accuracy base on the prediction done on initial 30% validation set
    val accuracy_cv = evaluator.evaluate(predictions_cv)
    println("*******Validation set accuracy = %.3f".format(accuracy_cv))

    //calculating the model accuracy base on the prediction done on initial 10% testData set
    val accuracy = evaluator.evaluate(predictions)
    println("*******Best accuracy from 10 percent initial testData  = %.3f".format(accuracy))


    //-------------- c- Sauvegarder le modèle entraîné pour pouvoir le réutiliser plus tard.------------------

    cvModel.write.overwrite().save("/home/skakeuh/Desktop/SPARK/tp_spark_Model")

  }


}