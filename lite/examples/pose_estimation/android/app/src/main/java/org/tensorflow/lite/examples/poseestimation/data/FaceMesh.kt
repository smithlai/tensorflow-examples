package org.tensorflow.lite.examples.poseestimation.data

data class FaceMesh(var keypoints:List<Triple<Float,Float,Float>>, var confidence: Float)