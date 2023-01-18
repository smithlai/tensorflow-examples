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
import org.tensorflow.lite.examples.poseestimation.data.DetectedObject
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.FaceMesh
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import kotlin.math.exp

abstract class ObjectDetector(
    private val interpreter: Interpreter,
    private var gpuDelegate: GpuDelegate?): AbstractDetector<List<DetectedObject>> {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val THRESHOLD = 0.7

        private const val TAG = "OD"

    }
    private var lastInferenceTimeNanos: Long = -1
    private val inputWidth = interpreter.getInputTensor(0).shape()[1]
    private val inputHeight = interpreter.getInputTensor(0).shape()[2]
    private var cropHeight = 0f
    private var cropWidth = 0f
    private var cropSize = 0
    private var reversed_sigmoid: Lazy<Float> = lazy{
        reverseSigmoid(THRESHOLD.toDouble()).toFloat()
    }

    private val visualizationUtils:VisualizationUtils = VisualizationUtils()
    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): List<DetectedObject> {
        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val estimationStartTimeNanos = SystemClock.elapsedRealtimeNanos()
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


        // [1 * 10 * 4] contains location/bounding box
        val outputLocations = outputMap[0] as Array<Array<FloatArray>>
        // [1 * 10]contains class
        val outputClasses = outputMap[1] as Array<FloatArray>
        // [1 * 10] score
        val outputScores = outputMap[2] as Array<FloatArray>
        // [1] number detection
        val numDetections = outputMap[3] as FloatArray

        val postProcessingStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val detected_list = postProcessModelOuputs(bitmap, outputLocations, outputClasses, outputScores, numDetections)
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
        return detected_list
    }

    /**
     * Convert heatmaps and offsets output of Posenet into a list of keypoints
     */
    private fun postProcessModelOuputs(
        bitmap: Bitmap,
        outputLocations: Array<Array<FloatArray>>,
        outputClasses: Array<FloatArray>,
        outputScores: Array<FloatArray>,
        numDetections: FloatArray
    ): List<DetectedObject> {

        val object_list = mutableListOf<DetectedObject>()

        for (i in 0 until numDetections[0].toInt()){
            if (outputScores[0][i] < THRESHOLD){
                continue
            }
            var tlx = outputLocations[0][i][1] * bitmap.width
            var tly = outputLocations[0][i][0] * bitmap.height
            var brx = outputLocations[0][i][3] * bitmap.width
            var bry = outputLocations[0][i][2] * bitmap.height

            object_list.add(
                DetectedObject(
                tl = Pair(tlx,tly),
                br = Pair(brx,bry),
                detectclass = outputClasses[0][i],
                detectscore = outputScores[0][i]
            ))
        }
        return object_list.toList()
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
        val imageProcessor = ImageProcessor.Builder().apply {
//            add(ResizeWithCropOrPadOp(cropSize, cropSize))
            add(ResizeOp(inputWidth, inputHeight, ResizeOp.ResizeMethod.BILINEAR))
//            add(NormalizeOp(MEAN, STD))
        }.build()
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }

    /**
     * Initializes an outputMap of 1 * x * y  FloatArrays for the model processing to populate.
     */
    private fun initOutputMap(interpreter: Interpreter): HashMap<Int, Any> {
        val outputMap = HashMap<Int, Any>()

        // 1 * 10 * 4 contains location/bounding box
        val outputLocations = interpreter.getOutputTensor(0).shape()

        outputMap[0] = Array(outputLocations[0]) {
            Array(outputLocations[1]) {
                FloatArray(outputLocations[2])
            }
        }

        // 1 * 10 contains class
        val outputClasses = interpreter.getOutputTensor(1).shape()
        outputMap[1] = Array(outputClasses[0]) {
            FloatArray(outputClasses[1])
        }
        // 1 * 10 score
        val outputScores = interpreter.getOutputTensor(2).shape()
        outputMap[2] = Array(outputScores[0]) {
            FloatArray(outputScores[1])
        }

        // 1 number detection
        val numDetections = interpreter.getOutputTensor(3).shape()
        outputMap[3] = FloatArray(numDetections[0])
        return outputMap
    }
    override fun drawKeypoints(bitmap: Bitmap, results: List<DetectedObject> ):Bitmap {
        val outputBitmap = visualizationUtils.drawKeypoints(bitmap,results)
        return outputBitmap
    }

    /** Returns value within [0,1].   */
    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-x)))
    }
    //    https://stackoverflow.com/questions/10097891/inverse-logistic-function-reverse-sigmoid-function
    /** Reverse sigmoid   */
    private fun reverseSigmoid(x: Double): Double {
        return Math.log(THRESHOLD / (1.0 - THRESHOLD))
    }
    class VisualizationUtils {
        companion object {
            /** Radius of circle used to draw keypoints.  */
            const val CIRCLE_RADIUS = 6f

            /** Width of line used to connected two keypoints.  */
            const val LINE_WIDTH = 4f

            /** The text size of the person id that will be displayed when the tracker is available.  */
            private const val TEXT_SIZE = 20f

            /** Distance from person id to the nose keypoint.  */
            const val PERSON_ID_MARGIN = 6f
        }
        // Draw line and point indicate body pose
        fun drawKeypoints(
            output: Bitmap,
            results: List<DetectedObject>
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
                textSize = TEXT_SIZE
                color = Color.BLUE
                textAlign = Paint.Align.LEFT
            }
            val originalSizeCanvas = Canvas(output)
            results.forEach { detected_object ->
                originalSizeCanvas.drawText(
                    "ID:" + detected_object.detectclass.toInt().toString() + " (" + String.format("%.2f",detected_object.detectscore*100) + "%)",
                    detected_object.tl.first,
                    detected_object.tl.second,
                    paintText
                )
                originalSizeCanvas.drawLine(
                    detected_object.tl.first,
                    detected_object.tl.second,
                    detected_object.br.first,
                    detected_object.tl.second,
                    paintLine
                )
                originalSizeCanvas.drawLine(
                    detected_object.br.first,
                    detected_object.tl.second,
                    detected_object.br.first,
                    detected_object.br.second,
                    paintLine
                )
                originalSizeCanvas.drawLine(
                    detected_object.br.first,
                    detected_object.br.second,
                    detected_object.tl.first,
                    detected_object.br.second,
                    paintLine
                )
                originalSizeCanvas.drawLine(
                    detected_object.tl.first,
                    detected_object.br.second,
                    detected_object.tl.first,
                    detected_object.tl.second,
                    paintLine
                )
            }
            return output
        }
    }
}
