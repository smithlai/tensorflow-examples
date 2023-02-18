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
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.camera.CameraSource
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.DetectedObject
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import kotlin.math.max

abstract class AbstractPoseDetector(
    private val interpreter: Interpreter,
    private val interopt: Interpreter.Options) : AbstractDetector<List<Person>>() {
    companion object {
        const val TAG = "PoseDetector"
        private const val MIN_CONFIDENCE = .2f
    }
    override var inference_results: List<Person> = listOf()
    var classificationResult: List<Pair<String, Float>>? = null
    var classifier: PoseClassifier? = null
        private set

    public val visualizationUtils: VisualizationUtils = VisualizationUtils()
//    abstract fun inferenceImage(bitmap: Bitmap): List<Person>
//    abstract fun lastInferenceTimeNanos(): Long

    override fun requestInferenceImage(bitmap: Bitmap): List<Person>{
        inference_results = super.requestInferenceImage(bitmap)
        inference_results = inference_results.filter { it.score > MIN_CONFIDENCE }
        inference_results.let {
//            // if the model only returns one item, allow running the Pose classifier.
//            if (it.isNotEmpty()) {
//                classificationResult = classifier?.classify(it[0])
//                Log.e("aaa", classificationResult.toString())
//            }
        }
        return inference_results
    }
    override fun drawResultOnBitmap(bitmap: Bitmap): Bitmap {
        val outputBitmap = visualizationUtils.drawBodyKeypoints(
            bitmap,
            getResults(), true
        )
//        if (results.isNotEmpty()) {
//            listener?.onDetectedInfo(it[0].score, classificationResult)
//        }
        return outputBitmap
    }

    override fun close() {
        interpreter.close()
        interopt.delegates.forEach {
            it.close()
        }
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
            output: Bitmap,
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
