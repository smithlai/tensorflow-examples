package org.tensorflow.lite.examples.poseestimation

import org.tensorflow.lite.examples.poseestimation.ml.AbstractObjectDetector
import kotlin.math.exp

class Utils {
    companion object{
        /** Returns value within [0,1].   */
        fun sigmoid(x: Float): Float {
            return (1.0f / (1.0f + exp(-x)))
        }
        //    https://stackoverflow.com/questions/10097891/inverse-logistic-function-reverse-sigmoid-function
        /** Reverse sigmoid   */
        fun reverseSigmoid(x: Double): Double {
            return Math.log(x / (1.0 - x))
        }
    }
}