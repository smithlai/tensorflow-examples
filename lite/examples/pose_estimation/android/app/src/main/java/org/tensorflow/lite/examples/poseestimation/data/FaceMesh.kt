package org.tensorflow.lite.examples.poseestimation.data

data class FaceMesh(var keypoints:List<Triple<Float,Float,Float>>,var facecrop:FaceCrop, var confidence: Float){
    enum class FaceMeshPart(val position: Int) {
        TL(0),
        BR(1),
        LEFT_EYE(2),
        RIGHT_EYE(3),
        NOSE(4),
        MOUTH(5),
        LEFT_EYE_TRAGION(6),
        RIGHT_EYE_TRAGION(7);
    }
}