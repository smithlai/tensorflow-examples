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
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.PointF
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.VisualizationUtils
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.KeyPoint
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import kotlin.math.exp

class FaceMeshDetector(private val interpreter: Interpreter, private var gpuDelegate: GpuDelegate?) :
    PoseDetector(interpreter, gpuDelegate) {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val TAG = "FaceMesh"
//        https://google.github.io/mediapipe/solutions/models.html#face-mesh
        private const val MODEL_FILENAME = "face_landmark.tflite"

        fun create(context: Context, device: Device): FaceMeshDetector {
            val settings: Pair<Interpreter.Options, GpuDelegate?> = getOption(device)
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

    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): List<Person> {
        val estimationStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val inputArray = arrayOf(processInputImage(bitmap).tensorBuffer.buffer)
        Log.i(
            TAG,
            String.format(
                "Scaling to [-1,1] took %.2f ms",
                (SystemClock.elapsedRealtimeNanos() - estimationStartTimeNanos) / 1_000_000f
            )
        )

        val outputMap = initOutputMap(interpreter)

        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        Log.e(
            TAG,
            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
        )

        val coordinates = outputMap[0] as Array<Array<Array<FloatArray>>>
        val offsets = outputMap[1] as Array<Array<Array<FloatArray>>>

        val postProcessingStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val person = postProcessModelOuputs(bitmap, coordinates, offsets)
        Log.i(
            TAG,
            String.format(
                "Postprocessing took %.2f ms",
                (SystemClock.elapsedRealtimeNanos() - postProcessingStartTimeNanos) / 1_000_000f
            )
        )

        return listOf(person)
    }

    /**
     * Convert heatmaps and offsets output of Posenet into a list of keypoints
     */
    private fun postProcessModelOuputs(
        bitmap: Bitmap,
        heatmaps: Array<Array<Array<FloatArray>>>,
        offsets: Array<Array<Array<FloatArray>>>
    ): Person {
        // 1 * 1 * 1 * 1404
        val numKeypoints = heatmaps[0][0][0].size  / 3
        val dataKeypoints = heatmaps[0][0][0]

        // Finds the (x, y, z)
        val keypointPositions = Array(numKeypoints) { Triple(0.0f, 0.0f, 0.0f) }
        for (keypoint in 0 until numKeypoints) {
            val base = keypoint*3
            keypointPositions[keypoint] = Triple(dataKeypoints[base], dataKeypoints[base+1],dataKeypoints[base+2])
        }

        // Calculating the x and y coordinates of the keypoints with offset adjustment.
        keypointPositions.forEachIndexed { idx, position ->
            val positionX = position.first
            val positionY = position.second
            val positionZ = position.third

            val inputImageCoordinateX = positionX*bitmap.width / inputWidth.toFloat()
            val inputImageCoordinateY = positionY*bitmap.height / inputHeight.toFloat()
            val inputImageCoordinateZ = positionZ
            keypointPositions[idx] = Triple(inputImageCoordinateX,inputImageCoordinateY, inputImageCoordinateZ)
//            Log.e("aaaaa", keypointPositions[idx].toString())
        }

        val keypointList = mutableListOf<KeyPoint>()
        enumValues<BodyPart>().forEachIndexed { idx, it ->
            keypointList.add(
                KeyPoint(
                    it,
                    PointF(keypointPositions[idx].first, keypointPositions[idx].second),
                    100f
                )
            )
        }
        return Person(keyPoints = keypointList.toList(), score = 100.0f)
    }

    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos


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
            add(ResizeWithCropOrPadOp(cropSize, cropSize))
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

    override fun visualize(overlay: Canvas, bitmap: Bitmap, persons: List<Person> ) {
        val outputBitmap = VisualizationUtils.drawBodyKeypoints(
            bitmap,
            persons.filter { it.score > 0.2 }, true
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


}
