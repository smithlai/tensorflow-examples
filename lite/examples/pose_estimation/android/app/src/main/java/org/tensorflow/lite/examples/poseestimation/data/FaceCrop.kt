package org.tensorflow.lite.examples.poseestimation.data

data class FaceCrop(val data:List<Pair<Float,Float>>, val score: Float) {
    enum class FaceCropPart(val position: Int) {
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
