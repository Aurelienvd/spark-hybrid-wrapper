package org.apache.spark.ml.feature.wrapper.genetic

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.{Estimator, Model, Predictor}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.feature.wrapper.WrapperParams
import org.apache.spark.ml.feature.wrapper.genetic.HybridGAImpl._
import org.apache.spark.util.Utils

/**
  * Params for [[HybridGA]] and [[HybridGAModel]]
  */
private[genetic] trait HybridGAParams extends WrapperParams {

    final val k: IntParam = new IntParam(this, "k", "the number of features to select", ParamValidators.gt(0))
    setDefault(k -> 50)

    final val populationSize: IntParam = new IntParam(this, "populationSize", "The size of the population of the" +
            "candidate solutions", ParamValidators.gt(0))
    setDefault(populationSize -> 100)

    final val crossoverProbability: DoubleParam = new DoubleParam(this, "crossoverProbability", "Probability that a crossover occurs",
        ParamValidators.inRange(0.0, 1.0))
    setDefault(crossoverProbability -> 1.0)

    final val mutationRate: DoubleParam = new DoubleParam(this, "mutationRate", "Probability that a given chromosome mutates",
        ParamValidators.inRange(0.0, 1.0))
    setDefault(mutationRate -> 0.1)

    final val selectionPressure: DoubleParam = new DoubleParam(this, "selectionPressure", "Selection Pressure coefficient for the" +
            "roulette wheel selection. A higher value means stronger selection pressure, making fitter chromosomes more likely to be selected.")
    setDefault(selectionPressure -> 0.25)

    final val penalty: DoubleParam = new DoubleParam(this, "penalty", "Penalty added to the fitness function to penalize" +
            "solutions with an incorrect number of features (i.e. #current selected features != k")
    setDefault(penalty -> 0.5)

    final val numberOfGenerations = new IntParam(this, "numberOfGeneration", "Number of iteration of the GA", ParamValidators.gt(0))
    setDefault(numberOfGenerations -> 25000)

    final val populationUpdateType = new UpdateTypeParam(this, "populationUpdateType", "Population update type (steady-state or generational)")
    setDefault(populationUpdateType -> SteadyState)

    final val crossoverOperator = new CrossoverOperatorParam(this, "crossoverOperator", "Crossover operator that produces two offspring")
    setDefault(crossoverOperator -> OnePointCrossover())

    final val localSearchProcedure = new LocalSearchParam(this, "localSearchProcedure", "Local search procedure that improves offspring")
    setDefault(localSearchProcedure -> NoSearch())

    /** @group getParam */
    def getK: Int = $(k)

    /** @group getParam */
    def getPopulationSize: Int = $(populationSize)

    /** @group getParam */
    def getCrossoverProbability: Double = $(crossoverProbability)

    /** @group getParam */
    def getMutationRate: Double = $(mutationRate)

    /** @group getParam */
    def getSelectionPressure: Double = $(selectionPressure)

    /** @group getParam */
    def getPenalty: Double = $(penalty)

    /** @group getParam */
    def getNumberOfGenerations: Int = $(numberOfGenerations)

    /** @group getParam */
    def getPopulationUpdateType: UpdateType = $(populationUpdateType)

    /** @group getParam */
    def getCrossoverOperator: CrossoverOperator = $(crossoverOperator)

    /** @group getParam */
    def getLocalSearchProcedure: LocalSearch = $(localSearchProcedure)

}

class HybridGA (override val uid: String)
        extends Estimator[HybridGAModel] with HybridGAParams with MLWritable {

    def this() = this(Identifiable.randomUID("HybridGA"))

    def setFeaturesCol(value: String): this.type = set(featuresCol, value)

    def setOutputCol(value: String): this.type = set(outputCol, value)

    def setLabelCol(value: String): this.type = set(labelCol, value)

    def setMinimizingMetric(value: Boolean): this.type = set(minimizingMetric, value)

    def setK(value: Int): this.type = set(k, value)

    def setPopulationSize(value: Int): this.type = set(populationSize, value)

    def setCrossoverProbability(value: Double): this.type = set(crossoverProbability, value)

    def setMutationRate(value: Double): this.type = set(mutationRate, value)

    def setSelectionPressure(value: Double): this.type = set(selectionPressure, value)

    def setPenalty(value: Double): this.type = set(penalty, value)

    def setNumberOfGenerations(value: Int): this.type = set(numberOfGenerations, value)

    def setPopulationUpdateType(value: UpdateType): this.type = set(populationUpdateType, value)

    def setCrossoverOperator(value: CrossoverOperator): this.type = set(crossoverOperator, value)

    def setLocalSearchProcedure(value: LocalSearch): this.type = set(localSearchProcedure, value)

    private def buildILSEOKParams(dataset: Dataset[_]): ILSEOKParams = ILSEOKParams(dataset, getFeaturesCol, getOutputCol,
        getLabelCol, getPredictor, getEvaluator, getMinimizingMetric, getK, getPopulationSize, getCrossoverProbability, getMutationRate,
        getSelectionPressure, getPenalty, getNumberOfGenerations, getPopulationUpdateType, getCrossoverOperator, getLocalSearchProcedure)

    /**
      * Runs a hybrid genetic algorithm and returns a [[HybridGAModel]] that contains the selected features.
      * Calls an algorithm implemented in HybridGAImpl.scala
      */

    override def fit(dataset: Dataset[_]): HybridGAModel = {
        transformSchema(dataset.schema, logging = true)
        new HybridGAModel(uid, ilseok(buildILSEOKParams(dataset))).setParent(this)
                .setFeaturesCol($(featuresCol))
                .setLabelCol($(labelCol))
                .setOutputCol($(outputCol))
    }

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def copy(extra: ParamMap): HybridGA = defaultCopy(extra)

    override def write: MLWriter = new HybridGA.HybridGAWriter(this)
}

object HybridGA extends MLReadable[HybridGA] {

    final val PREDICTOR_PATH: String = "/predictor"
    final val EVALUATOR_PATH: String = "/evaluator"

    override def read: MLReader[HybridGA] = new HybridGAReader

    override def load(path: String): HybridGA = super.load(path)

    private[HybridGA] class HybridGAWriter(instance: HybridGA) extends MLWriter {

        override protected def saveImpl(path: String): Unit = {
            DefaultParamsWriter.saveMetadata(instance, path, sc)
            instance.getPredictor.write.save(path+PREDICTOR_PATH)
            instance.getEvaluator.write.save(path+EVALUATOR_PATH)
        }
    }

    private class HybridGAReader extends MLReader[HybridGA] {
        private val className = classOf[HybridGA].getName

        override def load(path: String): HybridGA = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
            val cls = Utils.classForName(metadata.className)
            val instance = cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
            DefaultParamsReader.getAndSetParams(instance, metadata)
            val hga = instance.asInstanceOf[HybridGA]

            val predictor = DefaultParamsReader.loadParamsInstance[Predictor[_,_,_] with MLWritable](path+PREDICTOR_PATH, sc)
            val evaluator = DefaultParamsReader.loadParamsInstance[Evaluator with MLWritable](path+EVALUATOR_PATH, sc)

            hga.setPredictor(predictor).setEvaluator(evaluator)
        }
    }
}

class HybridGAModel private[ml] (override val uid: String, private val selectedFeatures: Array[Int])
        extends Model[HybridGAModel] with HybridGAParams with MLWritable {

    import HybridGAModel._

    def setFeaturesCol(value: String): this.type = set(featuresCol, value)

    def setOutputCol(value: String): this.type = set(outputCol, value)

    def setLabelCol(value: String): this.type = set(labelCol, value)

    override def transform(dataset: Dataset[_]): DataFrame = {
        transformSchema(dataset.schema, logging = true)
        new VectorSlicer().setInputCol($(featuresCol)).setOutputCol($(outputCol)).setIndices(selectedFeatures).transform(dataset)
    }

    override def transformSchema(schema: StructType): StructType = {
        validateAndTransformSchema(schema)
    }

    override def copy(extra: ParamMap): HybridGAModel = {
        val copied = new HybridGAModel(uid, selectedFeatures)
        copyValues(copied, extra).setParent(parent)
    }

    override def write: MLWriter = new HybridGAModelWriter(this)
}

object HybridGAModel extends MLReadable[HybridGAModel] {

    private[HybridGAModel] class HybridGAModelWriter(instance: HybridGAModel) extends MLWriter {
        private case class Data(selectedFeatures: Seq[Int])

        override protected def saveImpl(path: String): Unit = {
            DefaultParamsWriter.saveMetadata(instance, path, sc)
            val data = Data(instance.selectedFeatures.toSeq)
            val dataPath = new Path(path, "data").toString
            sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
        }
    }

    private class HybridGAModelReader extends MLReader[HybridGAModel] {
        private val className = classOf[HybridGAModel].getName

        override def load(path: String): HybridGAModel = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
            val dataPath = new Path(path, "data").toString
            val data = sparkSession.read.parquet(dataPath).select("selectedFeatures").head()
            val selectedFeatures = data.getAs[Seq[Int]](0).toArray
            val model = new HybridGAModel(metadata.uid, selectedFeatures)
            DefaultParamsReader.getAndSetParams(model, metadata)
            model
        }
    }

    override def read: MLReader[HybridGAModel] = new HybridGAModelReader

    override def load(path: String): HybridGAModel = super.load(path)
}
