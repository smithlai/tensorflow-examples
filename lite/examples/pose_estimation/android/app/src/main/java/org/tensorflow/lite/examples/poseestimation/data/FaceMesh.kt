package org.tensorflow.lite.examples.poseestimation.data

data class FaceMesh(var keypoints:List<Triple<Float,Float,Float>> = listOf(),
                    var leftIrisMesh:IrisMesh = IrisMesh(),
                    var rightIrisMesh:IrisMesh = IrisMesh(),
                    var facecrop:FaceCrop = FaceCrop(),
                    var confidence: Float = 0.0f){

    companion object {
        val silhouette = listOf(
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        )

        val lipsUpperOuter = listOf(61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
        val lipsLowerOuter = listOf(146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
        val lipsUpperInner = listOf(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
        val lipsLowerInner = listOf(78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)

        val left = mapOf<String, List<Int>>(
            "EyeUpper0" to listOf(246, 161, 160, 159, 158, 157, 173),
            "EyeLower0" to listOf(33, 7, 163, 144, 145, 153, 154, 155, 133),
            "EyeUpper1" to listOf(247, 30, 29, 27, 28, 56, 190),
            "EyeLower1" to listOf(130, 25, 110, 24, 23, 22, 26, 112, 243),
            "EyeUpper2" to listOf(113, 225, 224, 223, 222, 221, 189),
            "EyeLower2" to listOf(226, 31, 228, 229, 230, 231, 232, 233, 244),
            "EyeLower3" to listOf(143, 111, 117, 118, 119, 120, 121, 128, 245),
            "EyebrowUpper" to listOf(156, 70, 63, 105, 66, 107, 55, 193),
            "EyebrowLower" to listOf(35, 124, 46, 53, 52, 65),
            "Cheek" to listOf(205)
        )
        val right = mapOf<String, List<Int>>(
            "EyeUpper0" to listOf(466, 388, 387, 386, 385, 384, 398),
            "EyeLower0" to listOf(263, 249, 390, 373, 374, 380, 381, 382, 362),
            "EyeUpper1" to listOf(467, 260, 259, 257, 258, 286, 414),
            "EyeLower1" to listOf(359, 255, 339, 254, 253, 252, 256, 341, 463),
            "EyeUpper2" to listOf(342, 445, 444, 443, 442, 441, 413),
            "EyeLower2" to listOf(446, 261, 448, 449, 450, 451, 452, 453, 464),
            "EyeLower3" to listOf(372, 340, 346, 347, 348, 349, 350, 357, 465),
            "EyebrowUpper" to listOf(383, 300, 293, 334, 296, 336, 285, 417),
            "EyebrowLower" to listOf(265, 353, 276, 283, 282, 295),
            "Cheek" to listOf(425)
        )


        val midwayBetweenEyes = listOf(168)

        val noseTip = listOf(1)
        val noseBottom = listOf(2)
        val noseRightCorner = listOf(98)
        val noseLeftCorner = listOf(327)



//        val leftEyeIris = listOf(468, 469, 470, 471, 472)
//        val rightEyeIris = listOf(473, 474, 475, 476, 477)
    }
}