package org.apache.spark.ml.feature.wrapper

import org.apache.spark.ml.Predictor
import org.apache.spark.ml.evaluation.{Evaluator, RegressionEvaluator}
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param.{BooleanParam, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.util.{MLWritable, SchemaUtils}
import org.apache.spark.sql.types.StructType

private[wrapper] trait WrapperParams extends Params with HasFeaturesCol with HasOutputCol with HasLabelCol {

    var predictor: Predictor[_,_,_] with MLWritable = new DecisionTreeRegressor()

    var evaluator: Evaluator with MLWritable = new RegressionEvaluator()

    final val minimizingMetric: BooleanParam = new BooleanParam(this, "minimizingMetric", "Indicates if the evaluator's metric" +
            "has to be minimize. Usually, when it is a regression problem, the evaluation metric is an error measure which has to be minimized. When it is" +
            "a classification problem, the metric is an accuracy measure which has to be maximized. This parameter is important has it guides the search.")
    setDefault(minimizingMetric -> true)

    def getPredictor: Predictor[_,_,_] with MLWritable = predictor

    def getEvaluator: Evaluator with MLWritable = evaluator

    def getMinimizingMetric: Boolean = $(minimizingMetric)

    def setPredictor(predictor: Predictor[_,_,_] with MLWritable): this.type = {
        this.predictor = predictor
        this
    }

    def setEvaluator(evaluator: Evaluator with MLWritable): this.type = {
        this.evaluator = evaluator
        this
    }

    /** Validates and transforms the input schema. */
    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
        SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
    }
}
