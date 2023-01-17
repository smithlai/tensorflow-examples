package org.tensorflow.lite.examples.poseestimation.data

data class DetectedObject(var tl:Pair<Float,Float>, var br:Pair<Float,Float>, var detectclass: Float, var detectscore: Float)