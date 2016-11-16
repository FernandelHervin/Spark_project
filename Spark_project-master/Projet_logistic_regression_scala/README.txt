TP SPARK EXOPLANET

AUTHORS : Binome KAKEUH FOSSO Sidoine / KASA AKPO


JOB SOURCE CODE

Description

all preprocessing and cleaning job has been done in TP 2_3 (not include in this report)
we will take the cleanfile in order to train a model.

How to run the program

Compilation

In terminal, run following command :

# change to local directory of build.sbt file
cd "/home/skakeuh/Desktop/SPARK/Projet_logistic_regression_scala/tp_spark/tp_spark/build.sbt"

# compilation command on a shell
>sbt assembly 

example : on our computer:

(skakeuh@skakeuh-hp-pavilion-dv7-notebook-pc:~/Desktop/SPARK/Projet_logistic_regression_scala/tp_spark/tp_spark$ > sbt assembly)

>./bin/spark-submit (Job submission , please set the good path according to where you will store the project in your computer)

example :on our computer :

(skakeuh@skakeuh-hp-pavilion-dv7-notebook-pc:~/spark-2.0.0-bin-hadoop2.6$ ./bin/spark-submit --class com.sparkProject.Job /home/skakeuh/Desktop/SPARK/Projet_logistic_regression_scala/tp_spark/tp_spark/target/scala-2.11/tp_spark-assembly-1.0.jar )


after we have perform cross validation on logistic regression, we have use evaluator function in order to calculate the accuracy of the best hyperparameters.

The best accuracy is very close to 0.961 (values > 0.95) for regularization coefficients between 10-6 and 1.
