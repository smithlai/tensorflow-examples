/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation.ml

import android.content.Context
import android.graphics.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.camera.CameraSource
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import kotlin.math.max

abstract class PoseDetector(
    private val interpreter: Interpreter,
    private var gpuDelegate: GpuDelegate?) : AbstractDetector<List<Person>> {
    companion object {
        private const val MIN_CONFIDENCE = .2f
    }
    var classifier: PoseClassifier? = null
        private set
    val visualizationUtils: VisualizationUtils = VisualizationUtils()
//    abstract fun inferenceImage(bitmap: Bitmap): List<Person>
//    abstract fun lastInferenceTimeNanos(): Long


    override fun visualize(overlay: Canvas, bitmap: Bitmap, persons: List<Person> ) {
        val outputBitmap = visualizationUtils.drawBodyKeypoints(
            bitmap,
            persons.filter { it.score > MIN_CONFIDENCE }, true
        )

        overlay?.let { canvas ->
            val screenWidth: Int
            val screenHeight: Int
            val left: Int
            val top: Int

            if (canvas.height > canvas.width) {
                val ratio = outputBitmap.height.toFloat() / outputBitmap.width
                screenWidth = canvas.width
                left = 0
                screenHeight = (canvas.width * ratio).toInt()
                top = (canvas.height - screenHeight) / 2
            } else {
                val ratio = outputBitmap.width.toFloat() / outputBitmap.height
                screenHeight = canvas.height
                top = 0
                screenWidth = (canvas.height * ratio).toInt()
                left = (canvas.width - screenWidth) / 2
            }
            val right: Int = left + screenWidth
            val bottom: Int = top + screenHeight

            canvas.drawBitmap(
                outputBitmap, Rect(0, 0, outputBitmap.width, outputBitmap.height),
                Rect(left, top, right, bottom), null
            )
        }
    }

    override fun close() {
        gpuDelegate?.close()
        interpreter.close()
        classifier?.close()
        classifier = null
    }

    fun setClassifier(classifier: PoseClassifier?) {
        if (this.classifier != null) {
            this.classifier?.close()
            this.classifier = null
        }
        this.classifier = classifier
    }

    class VisualizationUtils {
        companion object {
            /** Radius of circle used to draw keypoints.  */
            const val CIRCLE_RADIUS = 6f

            /** Width of line used to connected two keypoints.  */
            const val LINE_WIDTH = 4f

            /** The text size of the person id that will be displayed when the tracker is available.  */
            const val PERSON_ID_TEXT_SIZE = 30f

            /** Distance from person id to the nose keypoint.  */
            const val PERSON_ID_MARGIN = 6f

            /** Pair of keypoints to draw lines between.  */
            val bodyJoints = listOf(
                Pair(BodyPart.NOSE, BodyPart.LEFT_EYE),
                Pair(BodyPart.NOSE, BodyPart.RIGHT_EYE),
                Pair(BodyPart.LEFT_EYE, BodyPart.LEFT_EAR),
                Pair(BodyPart.RIGHT_EYE, BodyPart.RIGHT_EAR),
                Pair(BodyPart.NOSE, BodyPart.LEFT_SHOULDER),
                Pair(BodyPart.NOSE, BodyPart.RIGHT_SHOULDER),
                Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
                Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
                Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
                Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
                Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
                Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
                Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
                Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
                Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
                Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
                Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
                Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
            )
        }

        // Draw line and point indicate body pose
        fun drawBodyKeypoints(
            input: Bitmap,
            persons: List<Person>,
            isTrackerEnabled: Boolean = false
        ): Bitmap {
            val paintCircle = Paint().apply {
                strokeWidth = CIRCLE_RADIUS
                color = Color.RED
                style = Paint.Style.FILL
            }
            val paintLine = Paint().apply {
                strokeWidth = LINE_WIDTH
                color = Color.RED
                style = Paint.Style.STROKE
            }

            val paintText = Paint().apply {
                textSize = PERSON_ID_TEXT_SIZE
                color = Color.BLUE
                textAlign = Paint.Align.LEFT
            }

            val output = input.copy(Bitmap.Config.ARGB_8888, true)
            val originalSizeCanvas = Canvas(output)
            persons.forEach { person ->
                // draw person id if tracker is enable
                if (isTrackerEnabled) {
                    person.boundingBox?.let {
                        val personIdX = max(0f, it.left)
                        val personIdY = max(0f, it.top)

                        originalSizeCanvas.drawText(
                            person.id.toString(),
                            personIdX,
                            personIdY - PERSON_ID_MARGIN,
                            paintText
                        )
                        originalSizeCanvas.drawRect(it, paintLine)
                    }
                }
                bodyJoints.forEach {
                    val pointA = person.keyPoints[it.first.position].coordinate
                    val pointB = person.keyPoints[it.second.position].coordinate
                    originalSizeCanvas.drawLine(pointA.x, pointA.y, pointB.x, pointB.y, paintLine)
                }

                person.keyPoints.forEach { point ->
                    originalSizeCanvas.drawCircle(
                        point.coordinate.x,
                        point.coordinate.y,
                        CIRCLE_RADIUS,
                        paintCircle
                    )
                }
            }
            return output
        }
    }
}
