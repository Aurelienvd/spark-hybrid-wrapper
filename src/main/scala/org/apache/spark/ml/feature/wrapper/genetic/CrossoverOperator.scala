package org.apache.spark.ml.feature.wrapper.genetic

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.jsontype.NamedType
import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper

import scala.collection.mutable
import scala.util.Random

@JsonTypeInfo(
    use = JsonTypeInfo.Id.CLASS,
    include = JsonTypeInfo.As.PROPERTY,
    property = "type")
trait CrossoverOperator {
    def crossover(parent1: Chromosome, parent2: Chromosome): (Chromosome, Chromosome)
}

case class OnePointCrossover() extends CrossoverOperator {
    override def crossover(parent1: Chromosome, parent2: Chromosome): (Chromosome, Chromosome) = {
        val cutPoint = 1 + Random.nextInt(parent1.genes.length-1)
        val offspring1 = Chromosome(parent1, parent2, 0, parent1.genes.slice(0, cutPoint) ++ parent2.genes.slice(cutPoint, parent2.genes.length), -1, false)
        val offspring2 = Chromosome(parent1, parent2, 0, parent2.genes.slice(0, cutPoint) ++ parent1.genes.slice(cutPoint, parent1.genes.length), -1, false)
        (offspring1, offspring2)
    }
}

/**
  * The implementation of MPointCrossover is taken and adapted from apache.commons.math4.genetics at
  * https://github.com/apache/commons-math/blob/master/src/main/java/org/apache/commons/math4/genetics/NPointCrossover.java
  */
case class MPointCrossover(mPoints: Int) extends CrossoverOperator {

    private def condSwap(c1: Chromosome, c2: Chromosome, value: Int): (Chromosome, Chromosome) = {
        if (value%2 == 0){
            (c1, c2)
        } else {
            (c2, c1)
        }
    }

    override def crossover(parent1: Chromosome, parent2: Chromosome): (Chromosome, Chromosome) = {
        val offspring1 = Chromosome(parent1, parent2, 0, new Array[Byte](parent1.genes.length), -1, inPopulation = false)
        val offspring2 = Chromosome(parent1, parent2, 0, new Array[Byte](parent2.genes.length), -1, inPopulation = false)
        var (remainingPoints, lastIndex) = (mPoints, 0)
        for (_ <- 0 until mPoints){
            val (c1, c2) = condSwap(offspring1, offspring2, mPoints-remainingPoints)
            val cutPoint = 1 + lastIndex + Random.nextInt(parent1.genes.length - lastIndex - remainingPoints)
            for (j <- lastIndex until cutPoint){
                c1.genes(j) = parent1.genes(j)
                c2.genes(j) = parent2.genes(j)
            }
            lastIndex = cutPoint
            remainingPoints -= 1
            }
        val (c1, c2) = condSwap(offspring1, offspring2, mPoints-remainingPoints)
        for (j <- lastIndex until parent1.genes.length){
            c1.genes(j) = parent1.genes(j)
            c2.genes(j) = parent2.genes(j)
        }
        (offspring1, offspring2)
    }
}

case class HalfUniformCrossover() extends CrossoverOperator {

    private def hammingDistanceAndPositions(parent1: Chromosome, parent2: Chromosome): (Int, Array[Int]) = {
        parent1.genes.zip(parent2.genes).zipWithIndex.foldLeft(0, new Array[Int](0)){ case ((sum, positions), ((b1, b2), i)) =>
            if ((b1^b2) == 1){
                (sum+1, positions:+i)
            } else{
                (sum, positions)
            }
        }
    }

    override def crossover(parent1: Chromosome, parent2: Chromosome): (Chromosome, Chromosome) = {
        val (diffSum, diffPositions) = hammingDistanceAndPositions(parent1, parent2)
        val positionsToSwap = Random.shuffle(diffPositions.toList).take(diffSum/2)
        val offspring1 = Chromosome(parent1, parent2, 0, parent1.genes.map{identity}, -1, inPopulation = false)
        val offspring2 = Chromosome(parent1, parent2, 0, parent2.genes.map{identity}, -1, inPopulation = false)
        positionsToSwap.foreach {pos =>
            val tmp = offspring1.genes(pos)
            offspring1.genes(pos) = offspring2.genes(pos)
            offspring2.genes(pos) = tmp
        }
        (offspring1, offspring2)
    }
}

object CrossoverOperator {
    private final val mapper: ObjectMapper = new ObjectMapper() with ScalaObjectMapper
    private val registeredClass = new mutable.HashSet[String]()

    mapper.registerModule(DefaultScalaModule)
    mapper.registerSubtypes(new NamedType(classOf[OnePointCrossover], OnePointCrossover().getClass.getName))
    mapper.registerSubtypes(new NamedType(classOf[MPointCrossover], MPointCrossover(0).getClass.getName))
    mapper.registerSubtypes(new NamedType(classOf[HalfUniformCrossover], HalfUniformCrossover().getClass.getName))
    registeredClass.add(OnePointCrossover().getClass.getName)
    registeredClass.add(MPointCrossover(0).getClass.getName)
    registeredClass.add(HalfUniformCrossover().getClass.getName)

    def registerNewOperator[T](operator: CrossoverOperator, clazz: Class[T]): Unit = {
        registeredClass.add(operator.getClass.getName)
        mapper.registerSubtypes(new NamedType(clazz, operator.getClass.getName))
    }

    def render(operator: CrossoverOperator): String = {
        if (registeredClass.contains(operator.getClass.getName)) {
            mapper.writerWithDefaultPrettyPrinter().writeValueAsString(operator)
        } else {
            throw new IllegalArgumentException(s"Cannot render unregistered operator of type ${operator.getClass.getName}." +
                    s"Call CrossoverOperator.registerNewOperator before using the render function.")
        }
    }

    def parse(json: String): CrossoverOperator = mapper.readValue(json, classOf[CrossoverOperator])
}