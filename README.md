# spark-hybrid-wrapper
This package adapts an algorithm developed by IL-Seok Oh, Jin-Seon Lee and Byung-Ro Moon [1] to a distributed context.
It is essentially a genetic-based wrapper that performs feature selection on big datasets. It uses Apache
Spark ml library. Aside from the usual genetic parameters such as *population size* or *mutation rate*, this package 
proposes different crossover operators (currently 3), local search procedures (currently 0) and
population update (currently 2). The predictor and the feature subset
evaluator can also be modified dynamically.

## Example

```scala
import org.apache.spark.ml.feature.wrapper.genetic._
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val crossover = HalfUniformCrossover()
val localSearch = NoSearch()
val evaluator = new BinaryClassificationEvaluator()
val predictor = new DecisionTreeClassifier()

val hga = new HybridGA()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setOutputCol("selectedFeatures")
    .setK(10)
    .setPopulationSize(20)
    .setNumberOfGenerations(100)
    .setPopulationUpdateType(SteadyState)
    .setPredictor(predictor)
    .setMutationRate(0.1)
    .setCrossoverProbability(1.0)
    .setCrossoverOperator(crossover)
    .setSelectionPressure(0.5)
    .setPenalty(0.5)
    .setLocalSearchProcedure(localSearch)
    .setEvaluator(evaluator)
    .setMinimizingMetric(true)

val result = hga.fit(training).transform(test)
```

## Crossover Operators

There are currently three operators implemented. They can be found in **CrossoverOperator.scala**.

 - *One-Point crossover*: A crossover operator that cuts the parents in two.
 - *M-Point crossover*: A crossover operator that cuts the parents M times.
 - *Half-Uniform crossover*: A crossover operator based on an article by Larry J. Eshelman [2].

There are two possibilities to add a new crossover operator.

 1. Add a new case class in **CrossoverOperator.scala** then add `mapper.registerSubtypes` and `registeredClass.add` in the CrossoverOperator object (cf. **CrossoverOperator.scala**).

 2. Define a new class in a new file that implements the *CrossoverOperator* trait. Then, call *registerNewOperator* that can be found in **CrossoverOperator.scala**.

## Local Search Procedures

There are currently no local search procedures implemented. The only case class found in **LocalSearch.scala** is the *NoSearch* case class. This represents
an absence of local search. The addition of a new local search procedure is the same as above.

## Population Update

There are currently two types of population update policies. They can be found in **GAParams.scala**.
 - *Steady State*: Two offspring are created at each step of the algorithm (i.e. one generation = two offspring).
 - *Generational*: *population size* offsprings are created at each step of the algorithm.

It is currently not possible to add a new population update policy.

## References

[1] Oh, Il-Seok, Jin-Seon Lee, and Byung-Ro Moon. "Hybrid genetic algorithms for feature selection." 
IEEE Transactions on pattern analysis and machine intelligence 26.11 (2004): 1424-1437.

[2] Eshelman, Larry J. "The CHC adaptive search algorithm: How to have safe search when engaging in nontraditional genetic recombination." 
Foundations of genetic algorithms. Vol. 1. Elsevier, 1991. 265-283.


