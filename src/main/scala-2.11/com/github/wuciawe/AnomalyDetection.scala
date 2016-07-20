package com.github.wuciawe

import org.apache.commons.math3.distribution.TDistribution
import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object AnomalyDetection {
  sealed trait Tail
  case object UpperTail extends Tail
  case object LowerTail extends Tail
  case object BothTails extends Tail

  case class Parameter(k: Double = 0.49, alpha: Double = 0.05, tail: Tail = BothTails)

  def detect[T : ClassTag](raw: Array[(T, Double)], param: Parameter): Array[T] = {
    val n = raw.length

    // Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    val maxOutliers = (n * param.k).toInt
    val anoms = ArrayBuffer.empty[T]

    // Compute test statistic until r=max_outliers values have been
    // removed from the sample.

    var data = raw.clone()
    var continue = true
    var i = 1
    while(i <= maxOutliers && continue) {
      val (median, sigma) = median_sigma(data.map(_._2))

      // protect against constant time series
      continue = sigma > 0

      if(continue) {
        val ares = param.tail match {
          case UpperTail => data.map(c => (c._1, (c._2 - median) / sigma))
          case LowerTail => data.map(c => (c._1, (median - c._2) / sigma))
          case BothTails => data.map(c => (c._1, math.abs(c._2 - median) / sigma))
        }

        val (r_i, r) = ares.maxBy(_._2)

        data = data.filterNot(_._1 == r_i)

        // Compute critical value.
        val prob = param.tail match {
          case BothTails => 1 - param.alpha/(2*(n-i+1))
          case _ => 1 - param.alpha/(n-i+1)
        }

        val td = new TDistribution(n - i - 1, 0.001)
        val t = td.inverseCumulativeProbability(prob)

        val lam = t * (n - i) / math.sqrt((n - i + 1 + math.pow(t, 2)) * (n - i + 1))

        continue = r > lam
        if(continue)
          anoms.append(r_i)
      }

      i += 1
    }
    anoms.toArray
  }

  case class ArrayView(arr: Array[Double], from: Int, until: Int) {
    def apply(n: Int) =
      if (from + n < until) arr(from + n)
      else throw new ArrayIndexOutOfBoundsException(n)

    def partitionInPlace(p: Double => Boolean): (ArrayView, ArrayView) = {
      var upper = until - 1
      var lower = from
      while (lower < upper) {
        while (lower < until && p(arr(lower))) lower += 1
        while (upper >= from && !p(arr(upper))) upper -= 1
        if (lower < upper) { val tmp = arr(lower); arr(lower) = arr(upper); arr(upper) = tmp }
      }
      (copy(until = lower), copy(from = lower))
    }

    def size = until - from
    def isEmpty = size <= 0
  }

  object ArrayView {
    def apply(arr: Array[Double]) = new ArrayView(arr, 0, arr.length)
  }

  @tailrec def findKInPlace(arr: ArrayView, k: Int)(implicit choosePivot: ArrayView => Double): Double = {
    val a = choosePivot(arr)
    val (s, b) = arr partitionInPlace (a > _)
    if (s.size == k) a
    // The following test is used to avoid infinite repetition
    else if (s.isEmpty) {
      val (s, b) = arr partitionInPlace (a == _)
      if (s.size > k) a
      else findKInPlace(b, k - s.size)
    } else if (s.size < k) findKInPlace(b, k - s.size)
    else findKInPlace(s, k)
  }

  def medianInPlace(arr: Array[Double])(implicit choosePivot: ArrayView => Double): Double =
    if(arr.length % 2 == 0) (findKInPlace(ArrayView(arr), (arr.length - 1) / 2) + findKInPlace(ArrayView(arr), arr.length / 2)) / 2
    else findKInPlace(ArrayView(arr), (arr.length - 1) / 2)

  def median_sigma(data: Array[Double]): (Double, Double) = {
    import scala.language.implicitConversions
    implicit def chooseRandomPivotInPlace(arr: ArrayView): Double = arr(scala.util.Random.nextInt(arr.size))
    val median = medianInPlace(data)
    (median, 1.4826 * medianInPlace(data.map(e => math.abs(e - median))))
  }
}