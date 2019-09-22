package org.apache.spark.ml.feature.wrapper.genetic

import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.jsontype.NamedType
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import org.apache.spark.sql.Dataset

import scala.collection.mutable

@JsonTypeInfo(
    use = JsonTypeInfo.Id.CLASS,
    include = JsonTypeInfo.As.PROPERTY,
    property = "type")
trait LocalSearch {
    def search(dataset: Dataset[_], featuresCol: String, labelCol: String, chromosome: Chromosome): Chromosome
}

case class NoSearch() extends LocalSearch {
    override def search(dataset: Dataset[_], featuresCol: String, labelCol: String, chromosome: Chromosome): Chromosome = chromosome
}

object LocalSearch {
    private final val mapper: ObjectMapper = new ObjectMapper() with ScalaObjectMapper
    private val registeredClass = new mutable.HashSet[String]()

    mapper.registerModule(DefaultScalaModule)
    mapper.registerSubtypes(new NamedType(classOf[NoSearch], NoSearch().getClass.getName))
    registeredClass.add(NoSearch().getClass.getName)

    def registerNewLocalSearch[T](search: LocalSearch, clazz: Class[T]): Unit = {
        registeredClass.add(search.getClass.getName)
        mapper.registerSubtypes(new NamedType(clazz, search.getClass.getName))
    }

    def render(search: LocalSearch): String = {
        if (registeredClass.contains(search.getClass.getName)) {
            mapper.writerWithDefaultPrettyPrinter().writeValueAsString(search)
        } else {
            throw new IllegalArgumentException(s"Cannot render unregistered operator of type ${search.getClass.getName}." +
                    s"Call LocalSearch.registerNewLocalSearch before using the render function.")
        }
    }

    def parse(json: String): LocalSearch = mapper.readValue(json, classOf[LocalSearch])
}