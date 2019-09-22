package org.apache.spark.ml.feature.wrapper.genetic

import java.util.concurrent.ThreadLocalRandom

import org.apache.spark.ml.{Pipeline, PipelineStage, PredictionModel, Predictor}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.feature.wrapper.Utils.binarySearchUpperBound

import scala.util.Random

/**
  * This file contains the implementation of the hybrid genetic algorithm defined by Il-Seok Oh et al. in their
  * 2004 article. The difference between the HybridGA class and this one is that HybridGA implements all the required structure
  * for the algorithm to properly run with Spark's mllib (making it usable in Pipelines etc..).
  * Here the "real" algorithm is implemented. The ilseok function is called inside [[HybridGA.fit]].
  */

private [genetic] case class ILSEOKParams(dataset: Dataset[_],
                                          featuresCol: String,
                                          outputCol: String,
                                          labelCol: String,
                                          predictor: PipelineStage,
                                          evaluator: Evaluator,
                                          minimizingMetric: Boolean,
                                          k: Int,
                                          populationSize: Int,
                                          crossoverProbability: Double,
                                          mutationRate: Double,
                                          selectionPressure: Double,
                                          penalty: Double,
                                          numberOfGenerations: Int,
                                          populationUpdateType: UpdateType,
                                          crossoverOperator: CrossoverOperator,
                                          localSearch: LocalSearch)

private [genetic] object HybridGAImpl {

    private def initPopulation(populationSize: Int, totalFeatures: Int, k: Int): Array[Chromosome] = {
        val population = Array.fill(populationSize)(Chromosome(null, null, 0, new Array[Byte](totalFeatures), -1, inPopulation = true))
        for (i <- 0 until populationSize){
            val chrom = population(i)
            chrom.positionInPopulation = i
            for (j <- 0 until totalFeatures){
                if (Random.nextDouble() < k.toDouble/totalFeatures) chrom.genes(j) = 1 else chrom.genes(j) = 0
            }
        }
        population
    }

    /**
      * Repairs a zero-filled genes by randomly flipping one bit.
      * This is done to prevent VectorSlicer to throw an IllegalArgumentException.
      *
      * @param genes
      * @return number of repaired genes.
      */
    private def repairGenes(genes: Array[Byte]): Int = {
        val pos = Random.nextInt(genes.length)
        genes(pos) = 1
        1
    }

    // Takes an array of zeroes and ones and returns a new array containing the indices of the 1s.
    private def genesToIndices(genes: Array[Byte]): Array[Int] = {
        var nonZeroes = 0
        genes.foreach { gene =>
            if (gene != 0) {
                nonZeroes += 1
            }
        }

        if (nonZeroes == 0){
            val addedZeroes = repairGenes(genes)
            nonZeroes += addedZeroes
        }

        val indices = new Array[Int](nonZeroes)
        var index = 0
        genes.zipWithIndex.foreach { case (gene, i) =>
            if (gene != 0){
                indices(index) = i
                index += 1
            }
        }
        indices
    }

    private def rouletteWheelSelection(population: Array[Chromosome], selectionPressure: Double): (Chromosome, Chromosome) = {
        val sortedPopulation = population.sortWith{_.fitness > _.fitness}
        val cumulativeProb = sortedPopulation.zipWithIndex.scanLeft(0.0){ case (acc, (chrom, i)) =>
                acc + selectionPressure*math.pow(1-selectionPressure, i)
        }.drop(1)

        val r = ThreadLocalRandom.current().nextDouble(0, cumulativeProb.last)
        val parent1 = sortedPopulation(binarySearchUpperBound(cumulativeProb, r).get)
        var parent2: Option[Chromosome] = None
        while (parent2.isEmpty) {
            val r = ThreadLocalRandom.current().nextDouble(0, cumulativeProb.last)
            val optParent = sortedPopulation(binarySearchUpperBound(cumulativeProb, r).get)
            if (optParent != parent1){
                parent2 = Some(optParent)
            }
        }
        (parent1, parent2.get)
    }

    // Removes null chromosomes from offsprings (happens when crossoverProbability < 1 and updateType = Generational).
    private def compressOffsprings(offspring: Array[Chromosome]): Array[Chromosome] = offspring.filterNot(c => c == null)

    private def crossoverOccurs(updateType: UpdateType, crossoverProbability: Double): Boolean = updateType match {
        case SteadyState => true
        case Generational =>
            val r = Random.nextDouble()
            r < crossoverProbability
    }

    private def computeMutationProbabilities(chrom1: Chromosome, chrom2: Chromosome, rate: Double): (Double, Double, Double) = {
        val (chrom1Num1, chrom2Num1) = (chrom1.genes.sum, chrom2.genes.sum)
        val (chrom1Num0, chrom2Num0) = (chrom1.genes.length-chrom1Num1, chrom2.genes.length-chrom2Num1)
        (rate, rate*(chrom1Num1.toDouble/math.max(chrom1Num0, 1)), rate*(chrom2Num1.toDouble/math.max(chrom2Num0,1)))
    }

    private def mutateChromosome(chrom: Chromosome, p1: Double, p0: Double): Chromosome = {
        for (i <- chrom.genes.indices){
            val r = Random.nextDouble()
            if (chrom.genes(i) == 1 && r < p1){
                chrom.genes(i) = 0
            } else if (chrom.genes(i) == 0 && r < p0){
                chrom.genes(i) = 1
            }
        }
        chrom
    }

    private def mutate(chromosomes: (Chromosome, Chromosome), rate: Double): (Chromosome, Chromosome) = {
        val (p1, chrom1p0, chrom2p0) = computeMutationProbabilities(chromosomes._1, chromosomes._2, rate)
        (mutateChromosome(chromosomes._1, p1, chrom1p0), mutateChromosome(chromosomes._2, p1, chrom2p0))
    }

    private def offspringCanReplace(offspring: Chromosome, parent: Chromosome): Boolean = {
        if (parent.inPopulation){
            offspring.fitter(parent)
        }
        false
    }

    private def replaceInPopulation(replacingChrom: Chromosome, replacedChrom: Chromosome, population: Array[Chromosome]): Unit = {
        population(replacedChrom.positionInPopulation) = replacingChrom
        replacingChrom.placeInPopulation(replacedChrom.positionInPopulation)
        replacedChrom.removeFromPopulation()
    }

    private def updatePopulation(population: Array[Chromosome], offsprings: Array[Chromosome]): Unit = {
        offsprings.foreach{ offspring =>
            val (mostSimilarParent, lessSimilarParent) = offspring.getMostAndLessSimilarParents
            if (offspringCanReplace(offspring, mostSimilarParent)){
                replaceInPopulation(offspring, mostSimilarParent, population)
            } else if (offspringCanReplace(offspring, lessSimilarParent)){
                replaceInPopulation(offspring, lessSimilarParent, population)
            } else {
                val sortedPopulation = population.sortWith{_.fitness > _.fitness}
                replaceInPopulation(offspring, sortedPopulation.last, population)
            }
        }
    }

    // Computes a penalty factor by increasing the penalty for genes with a number of 1s that differs from k.
    private def computePenalty(penalty: Double, numberOfFeaturesInGenes: Int, k: Int): Double = penalty*math.abs(numberOfFeaturesInGenes-k)

    private def evaluatePopulation(dataset: Dataset[_], featuresCol: String, labelCol: String, population: Array[Chromosome],
                                   penalty: Double, k: Int, predictor: PipelineStage, evaluator: Evaluator, sign: Int): Unit = {
        population.foreach { chrom =>
            val indices = genesToIndices(chrom.genes)
            val reducedDataset = new VectorSlicer().setInputCol(featuresCol).setOutputCol("slicedFeatures").setIndices(indices).transform(dataset)
            val Array(training, test) = reducedDataset.randomSplit(Array(0.75, 0.25))
            val stage = predictor.set(predictor.getParam("featuresCol"), "slicedFeatures").set(predictor.getParam("labelCol"), labelCol)
            val model = new Pipeline().setStages(Array(stage)).fit(training)
            val predictions = model.transform(test)
            chrom.fitness = sign * evaluator.evaluate(predictions) - computePenalty(penalty, indices.length, k)
        }
    }

    def ilseok(params: ILSEOKParams): Array[Int] = {
        val totalFeatures: Int = params.dataset.select(params.featuresCol).first.getAs[Vector](0).size
        val sign = if (params.minimizingMetric) -1 else 1
        require(params.k <= totalFeatures, s"Total number of features ($totalFeatures) must not be smaller than k=${params.k}")

        val population = initPopulation(params.populationSize, totalFeatures, params.k)
        evaluatePopulation(params.dataset, params.featuresCol, params.labelCol, population, params.penalty, params.k,
            params.predictor, params.evaluator, sign)

        for (_ <- 0 until params.numberOfGenerations) {
            val (offsprings, loopEnd) = params.populationUpdateType match {
                case SteadyState => (new Array[Chromosome](2), 1)
                case Generational => (new Array[Chromosome](params.populationSize), params.populationSize)
            }
            for (i <- 0 until loopEnd by 2){
                val (parent1, parent2) = rouletteWheelSelection(population, params.selectionPressure)
                if (crossoverOccurs(params.populationUpdateType, params.crossoverProbability)){
                    val (offspring1, offspring2) = mutate(params.crossoverOperator.crossover(parent1, parent2), params.mutationRate)
                    offsprings(i) = params.localSearch.search(params.dataset, params.featuresCol, params.labelCol, offspring1)
                    offsprings(i+1) = params.localSearch.search(params.dataset, params.featuresCol, params.labelCol, offspring2)
                }
            }
            val coffsprings = compressOffsprings(offsprings)
            evaluatePopulation(params.dataset, params.featuresCol, params.labelCol, coffsprings, params.penalty,
                params.k, params.predictor, params.evaluator, sign)
            updatePopulation(population, coffsprings)
        }
        val bestChromosome = population.sortWith{_.fitness > _.fitness}.head
        genesToIndices(bestChromosome.genes)
    }
}
