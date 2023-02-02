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
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.FaceCrop
import org.tensorflow.lite.examples.poseestimation.data.FaceMesh
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.lang.Float.min
import kotlin.math.exp

class FaceMeshDetector(
    private val interpreter: Interpreter,
    private var gpuDelegate: GpuDelegate?): AbstractDetector<List<FaceMesh>> {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val THRESHOLD = 0.90f
        private const val STROKE_RATIO = 120
        private const val TAG = "FaceMesh"
//        https://google.github.io/mediapipe/solutions/models.html#face-mesh
        private const val MODEL_FILENAME = "face_landmark.tflite"
//        private const val MODEL_FILENAME = "face_landmark_with_attention.tflite"

        fun create(context: Context, device: Device): FaceMeshDetector {
            val settings: Pair<Interpreter.Options, GpuDelegate?> = AbstractDetector.getOption(device)
            val options = settings.first
            var gpuDelegate = settings.second

            return FaceMeshDetector(
                Interpreter(
                    FileUtil.loadMappedFile(
                        context,
                        MODEL_FILENAME
                    ), options
                ),
                gpuDelegate
            )
        }
    }

    private var lastInferenceTimeNanos: Long = -1
    private val inputWidth = interpreter.getInputTensor(0).shape()[1]
    private val inputHeight = interpreter.getInputTensor(0).shape()[2]
    private var cropHeight = 0f
    private var cropWidth = 0f
    private var cropSize = 0
    private val visualizationUtils:VisualizationUtils = VisualizationUtils()
    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): List<FaceMesh> {
        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
//        val estimationStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val inputArray = arrayOf(processInputImage(bitmap).tensorBuffer.buffer)
//        Log.i(
//            TAG,
//            String.format(
//                "Scaling to [-1,1] took %.2f ms",
//                (SystemClock.elapsedRealtimeNanos() - estimationStartTimeNanos) / 1_000_000f
//            )
//        )

        val outputMap = initOutputMap(interpreter)

//        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
//        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
//        Log.e(
//            TAG,
//            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
//        )

        val landmark_buffer = outputMap[0] as Array<Array<Array<FloatArray>>>
        val faceflag_buffer = outputMap[1] as Array<Array<Array<FloatArray>>>

        val postProcessingStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val facemeshes = postProcessModelOuputs(bitmap, landmark_buffer, faceflag_buffer)
//        Log.i(
//            TAG,
//            String.format(
//                "Postprocessing took %.2f ms",
//                (SystemClock.elapsedRealtimeNanos() - postProcessingStartTimeNanos) / 1_000_000f
//            )
//        )
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        Log.i(
            TAG,
            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
        )
        return facemeshes
    }

    /**
     * Convert heatmaps and offsets output of Posenet into a list of keypoints
     */
    private fun postProcessModelOuputs(
        bitmap: Bitmap,
        landmark_buffer: Array<Array<Array<FloatArray>>>,
        faceflag_buffer: Array<Array<Array<FloatArray>>>
    ): List<FaceMesh> {
        // 1 * 1 * 1 * 1404
        val numKeypoints = landmark_buffer[0][0][0].size  / 3
        val pos_buffer_p = landmark_buffer[0][0][0]

        // Finds the (x, y, z)
        val keypointPositions = Array(numKeypoints) { Triple(0.0f, 0.0f, 0.0f) }
        for (keypoint in 0 until numKeypoints) {
            val base = keypoint*3
            keypointPositions[keypoint] = Triple(pos_buffer_p[base], pos_buffer_p[base+1],pos_buffer_p[base+2])
        }

        // Calculating the x and y coordinates of the keypoints with offset adjustment.
        val keypointPositions2 = Array(numKeypoints) { Triple(0f, 0f, 0f) }
        keypointPositions.forEachIndexed { idx, position ->
            val positionX = position.first
            val positionY = position.second
            val positionZ = position.third

            val inputImageCoordinateX = (positionX*bitmap.width / inputWidth).toFloat()
            val inputImageCoordinateY = (positionY*bitmap.height / inputHeight).toFloat()
            val inputImageCoordinateZ = positionZ.toFloat()
            keypointPositions2[idx] = Triple(inputImageCoordinateX,inputImageCoordinateY, inputImageCoordinateZ)
//            Log.e("aaaaa", keypointPositions[idx].toString())
        }
        val faceflag = faceflag_buffer[0][0][0][0]
        val confidence = sigmoid(faceflag)
        var facelist = mutableListOf<FaceMesh>()
        if (confidence>= THRESHOLD) {
            val face_crop = FaceCrop(
                listOf(Pair(0f, 0f), Pair(bitmap.width.toFloat(), bitmap.height.toFloat())),
                1.0f
            )
            facelist.add(
                FaceMesh(
                    keypoints = keypointPositions2.toList(),
                    face_crop,
                    confidence = confidence
                )
            )
        }
        return facelist.toList()
    }

    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos
    override fun close() {
        gpuDelegate?.close()
        interpreter.close()
    }

    /**
     * Scale and crop the input image to a TensorImage.
     */
    private fun processInputImage(bitmap: Bitmap): TensorImage {
        // reset crop width and height
        cropWidth = 0f
        cropHeight = 0f
        cropSize = if (bitmap.width > bitmap.height) {
            cropHeight = (bitmap.width - bitmap.height).toFloat()
            bitmap.width
        } else {
            cropWidth = (bitmap.height - bitmap.width).toFloat()
            bitmap.height
        }
//        Log.e("aaaaa", inputWidth.toString() + " " + inputHeight.toString())
        val imageProcessor = ImageProcessor.Builder().apply {
//            add(ResizeWithCropOrPadOp(cropSize, cropSize))
            add(ResizeOp(inputWidth, inputHeight, ResizeOp.ResizeMethod.BILINEAR))
            add(NormalizeOp(MEAN, STD))
        }.build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }

    /**
     * Initializes an outputMap of 1 * x * y * z FloatArrays for the model processing to populate.
     */
    private fun initOutputMap(interpreter: Interpreter): HashMap<Int, Any> {
        val outputMap = HashMap<Int, Any>()

        // 1 * 1 * 1 * 1404 contains heatmaps

        val coordinations_shape = interpreter.getOutputTensor(0).shape()

        outputMap[0] = Array(coordinations_shape[0]) {
            Array(coordinations_shape[1]) {
                Array(coordinations_shape[2]) {
                    FloatArray(coordinations_shape[3])
                }
            }
        }

        // 1 * 1 * 1 * 1 contains offsets
        val offest_shape = interpreter.getOutputTensor(1).shape()
        outputMap[1] = Array(offest_shape[0]) {
            Array(offest_shape[1]) {
                Array(offest_shape[2]) {
                    FloatArray(offest_shape[3])
                }
            }
        }
        return outputMap
    }
    override fun drawKeypoints(bitmap: Bitmap, results: List<FaceMesh> ): Bitmap {
        val outputBitmap = visualizationUtils.drawKeypoints(bitmap,results)
        return outputBitmap
    }
    /** Returns value within [0,1].   */
    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-x)))
    }
    class VisualizationUtils {
        companion object {
            /** Radius of circle used to draw keypoints.  */
            const val CIRCLE_RADIUS = 6f

            /** Width of line used to connected two keypoints.  */
            const val LINE_WIDTH = 4f

            /** The text size of the person id that will be displayed when the tracker is available.  */
            private const val PERSON_ID_TEXT_SIZE = 30f

            /** Distance from person id to the nose keypoint.  */
            const val PERSON_ID_MARGIN = 6f
        }
        // Draw line and point indicate body pose
        fun drawKeypoints(
            output: Bitmap,
            results: List<FaceMesh>
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
            results.forEach { facemesh ->
                val tl = facemesh.facecrop.data.get(0)
                val x1 = tl.first
                val y1 = tl.second
                val br = facemesh.facecrop.data.get(1)
                val x2 = br.first
                val y2 = br.second
                val w = x2- x1
                val h = y2- y1
                val circle_rad = min(w,h)/STROKE_RATIO
                paintCircle.strokeWidth = circle_rad
                facemesh.keypoints.forEach{ keypoint->
//                Log.e("aaaaa", keypoint.toString())
                    originalSizeCanvas.drawCircle(
                        keypoint.first+x1,
                        keypoint.second+y1,
                        circle_rad,
                        paintCircle
                    )
                }
            }
            return output
        }
    }
}
