package org.apache.spark.ml.feature.wrapper.genetic

private[genetic] class Chromosome(var parent1: Chromosome,
                                  var parent2: Chromosome,
                                  var fitness: Double,
                                  val genes: Array[Byte],
                                  var positionInPopulation: Int,
                                  var inPopulation: Boolean) {
    def fitter(other: Chromosome): Boolean = this.fitness > other.fitness

    def getMostAndLessSimilarParents: (Chromosome, Chromosome) = {
        val hammingDistance1 = this.genes.zip(parent1.genes).map {case (b1, b2) => b1^b2}.sum
        val hammingDistance2 = this.genes.zip(parent2.genes).map {case (b1, b2) => b1^b2}.sum
        if (hammingDistance1 <= hammingDistance2){
            (parent1, parent2)
        } else {
            (parent2, parent1)
        }
    }

    def placeInPopulation(indexInPopulation: Int): Unit = {
        this.inPopulation = true
        this.positionInPopulation = indexInPopulation
        this.parent1 = null
        this.parent2 = null
    }

    def removeFromPopulation(): Unit = {
        positionInPopulation = -1
        inPopulation = false
    }
}

private[genetic] object Chromosome {
    def apply(parent1: Chromosome, parent2: Chromosome, fitness: Double, genes: Array[Byte], positionInPopulation: Int, inPopulation: Boolean): Chromosome =
        new Chromosome(parent1, parent2, fitness, genes, positionInPopulation, inPopulation)
}