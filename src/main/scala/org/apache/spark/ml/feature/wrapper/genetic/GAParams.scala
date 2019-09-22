package org.apache.spark.ml.feature.wrapper.genetic

import org.apache.spark.ml.param.{Param, ParamPair}
import org.apache.spark.ml.util.Identifiable
import org.json4s._
import org.json4s.jackson.JsonMethods._

sealed trait UpdateType
case object SteadyState extends UpdateType
case object Generational extends UpdateType

private[genetic] class UpdateTypeParam(parent: String, name: String, doc: String, isValid: UpdateType => Boolean)
        extends Param[UpdateType](parent, name, doc, isValid) {

    def this(parent: String, name: String, doc: String) = this(parent, name, doc, (_: UpdateType) => true)

    def this(parent: Identifiable, name: String, doc: String, isValid: UpdateType => Boolean) = this(parent.uid, name, doc, isValid)

    def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

    override def w(value: UpdateType): ParamPair[UpdateType] = super.w(value)

    override def jsonEncode(value: UpdateType): String = {
        value match {
            case SteadyState => compact(render(JString("SteadyState")))
            case Generational => compact(render(JString("Generational")))
        }
    }

    override def jsonDecode(json: String): UpdateType = {
        parse(json) match {
            case JString("SteadyState") => SteadyState
            case JString("Generational") => Generational
            case _ => throw new IllegalArgumentException(s"Cannot decode ${parse(json)} to UpdateType.")
        }
    }
}

private[genetic] class CrossoverOperatorParam(parent: String, name: String, doc: String, isValid: CrossoverOperator => Boolean)
        extends Param[CrossoverOperator](parent, name, doc, isValid) {

    def this(parent: String, name: String, doc: String) = this(parent, name, doc, (_ :CrossoverOperator) => true)

    def this(parent: Identifiable, name: String, doc: String, isValid: CrossoverOperator => Boolean) = this(parent.uid, name, doc, isValid)

    def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

    override def w(value: CrossoverOperator): ParamPair[CrossoverOperator] = super.w(value)

    override def jsonEncode(value: CrossoverOperator): String = CrossoverOperator.render(value)

    override def jsonDecode(json: String): CrossoverOperator = CrossoverOperator.parse(json)
}

private[genetic] class LocalSearchParam(parent: String, name: String, doc: String, isValid: LocalSearch => Boolean)
        extends Param[LocalSearch](parent, name, doc, isValid) {

    def this(parent: String, name: String, doc: String) = this(parent, name, doc, (_ :LocalSearch) => true)

    def this(parent: Identifiable, name: String, doc: String, isValid: LocalSearch => Boolean) = this(parent.uid, name, doc, isValid)

    def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

    override def w(value: LocalSearch): ParamPair[LocalSearch] = super.w(value)

    override def jsonEncode(value: LocalSearch): String = LocalSearch.render(value)

    override def jsonDecode(json: String): LocalSearch = LocalSearch.parse(json)
}
