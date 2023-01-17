package org.tensorflow.lite.examples.poseestimation.data

data class FaceCrop(var tl:Pair<Float,Float>,
                    var rb:Pair<Float,Float>,
                    var leye:Pair<Float,Float>,
                    var reye:Pair<Float,Float>,
                    var nose:Pair<Float,Float>,
                    var mouth:Pair<Float,Float>,
                    var leye_tragion:Pair<Float,Float>,
                    var reye_tragion:Pair<Float,Float>,
                    var confidence:Float){

    enum class FaceCropPart(val position: Int) {
        TL(0),
        BR(1),
        LEFT_EYE(2),
        RIGHT_EYE(3),
        NOSE(4),
        MOUTH(5),
        LEFT_EYE_TRAGION(6),
        RIGHT_EYE_TRAGION(7);

        companion object{
            val map = values().associateBy(FaceCropPart::position)

//            fun fromInt(position: Int): BodyPart = map.getValue(position)
        }
    }
}