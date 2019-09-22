package org.apache.spark.ml.feature.wrapper

object Utils {

    /**
      * Returns the index of the first upper bound ( the first value strictly > ) in an array of values.
      * E.g. given Array(10,25,50),
      * the first upper bound of 9 is 10, the first upper bound of 49 is 50. If key > values.last, returns None.
      *
      * Warning: binarySearchUpperBound(values, key = values.last) returns None.
      */
    def binarySearchUpperBound[T](values: Array[T], key: T)(implicit order: Ordering[T]): Option[Int] = {
        if (order.gteq(key, values.last)){
            None
        }
        else {
            var left = 0
            var right = values.length - 1
            while (left < right) {
                val middle = left + (right - left) / 2
                if (order.lt(key, values(middle))) {
                    right = middle
                } else {
                    left = middle+1
                }
            }
            Some(left)
        }
    }
}
