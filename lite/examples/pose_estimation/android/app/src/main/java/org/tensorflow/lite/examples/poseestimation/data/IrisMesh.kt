package org.tensorflow.lite.examples.poseestimation.data

data class IrisMesh(var output_eyes_contours_and_brows:List<Triple<Float,Float,Float>> = listOf(),
                    var output_iris:List<Triple<Float,Float,Float>> = listOf(),
                    var rect:List<Pair<Float,Float>> = listOf(Pair(0f,0f),Pair(0f,0f))
)