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
import androidx.core.graphics.green
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.Utils
import org.tensorflow.lite.examples.poseestimation.data.*
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.lang.Float.min
import kotlin.math.exp

class LeftIrisDetector(
    private val interpreter: Interpreter,
    private val interopt: Interpreter.Options): AbstractDetector<IrisMesh>() {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val THRESHOLD = 0.90f
        private const val STROKE_RATIO = 120
        private const val TAG = "IrisMesh"
//        https://google.github.io/mediapipe/solutions/models.html#face-mesh
        private const val MODEL_FILENAME = "iris_landmark.tflite"

        fun create(context: Context, device: Device): LeftIrisDetector {
            val options = AbstractDetector.getOption(device, context)

            return LeftIrisDetector(
                Interpreter(
                    FileUtil.loadMappedFile(
                        context,
                        MODEL_FILENAME
                    ), options
                ),
                options
            )
        }
    }

    override var inference_results: IrisMesh = IrisMesh()
    private var lastInferenceTimeNanos: Long = -1
    private val inputWidth = interpreter.getInputTensor(0).shape()[1]
    private val inputHeight = interpreter.getInputTensor(0).shape()[2]
    private var cropHeight = 0f
    private var cropWidth = 0f
    private var cropSize = 0
    private val visualizationUtils:VisualizationUtils = VisualizationUtils()
    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): IrisMesh {
        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
//        val estimationStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val inputArray = arrayOf(processInputImage(bitmap).tensorBuffer.buffer)

        val outputMap = initOutputMap(interpreter)

//        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
//        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
//        Log.e(
//            TAG,
//            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
//        )

//        val postProcessingStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val facemeshes = postProcessModelOuputs(bitmap, outputMap)

        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        Log.i(
            TAG,
            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
        )
        return facemeshes
    }

    private fun flatToDimScale(dim: Int, src: FloatArray,w1:Int,h1:Int,w2:Int,h2:Int): Array<Triple<Float,Float,Float>>{
        val dim_points = Array(src.size / dim) { Triple(0.0f, 0.0f, 0.0f) }
        for (pt_idx in 0 until dim_points.size) {
            val base = pt_idx*dim
            val positionX = src[base]
            val positionY = src[base+1]
            var positionZ = 0f
            if (dim > 2)
                positionZ = src[base+2]

            val inputImageCoordinateX = (positionX*w2 / w1).toFloat()
            val inputImageCoordinateY = (positionY*h2 / h1).toFloat()
            val inputImageCoordinateZ = positionZ.toFloat()
            dim_points[pt_idx] = Triple(inputImageCoordinateX,inputImageCoordinateY, inputImageCoordinateZ)
        }
        return dim_points
    }
    /**
     * Convert heatmaps and offsets output of Posenet into a list of keypoints
     */
    private fun postProcessModelOuputs(
        bitmap: Bitmap,
        outputMap: HashMap<Int,*>
    ): IrisMesh {
        var outputcount = 0

        // All landmarks
        val output_eyes_contours_and_brows_buffer = outputMap[outputcount] as Array<FloatArray>

        // output_eyes_contours_and_brows [1 * 213] =>  contains 71 * 3D points
        val output_eyes_contours_and_brows_buffer_p = output_eyes_contours_and_brows_buffer[0]
        val output_eyes_contours_and_brows = flatToDimScale(3, output_eyes_contours_and_brows_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)
//        Log.e("aaaa", inputWidth.toString() +" " + inputHeight +" " + bitmap.width +" " + bitmap.height)

        outputcount += 1
        val output_iris_buffer = outputMap[outputcount] as Array<FloatArray>
        val output_iris_buffer_p = output_iris_buffer[0]
        // output_iris [1 * 15] contains 5 * 3D pupil points
        val output_iris = flatToDimScale(3, output_iris_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)
        val iris = IrisMesh(
            output_eyes_contours_and_brows = output_eyes_contours_and_brows.toList(),
            output_iris = output_iris.toList(),
            rect = listOf(Pair(0f,0f), Pair(bitmap.width.toFloat(),bitmap.height.toFloat()))
        )
        return iris
    }

    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos
    override fun close() {
        interpreter.close()
        interopt.delegates.forEach {
            it.close()
        }
//        gpuDelegate?.close()
//        gpuDelegate = null
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
        var outputcount = 0
        val outputMap = HashMap<Int, Any>()
        // output_eyes_contours_and_brows [1 * 213] =>  contains 71 * 3D points
        val coordinations_shape = interpreter.getOutputTensor(outputcount).shape()

        outputMap[outputcount] = Array(coordinations_shape[0]) {
            FloatArray(coordinations_shape[1])
        }


        outputcount += 1
        // output_iris [1 * 15] contains 5 * 3D pupil points
        val score_shape = interpreter.getOutputTensor(outputcount).shape()
        outputMap[outputcount] = Array(score_shape[0]) {
            FloatArray(score_shape[1])
        }

        return outputMap
    }
    override fun drawResultOnBitmap(bitmap: Bitmap): Bitmap {
        val outputBitmap = visualizationUtils.drawKeypoints(bitmap, getResults())
        return outputBitmap
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
            results: IrisMesh
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

            val tl = results.rect.get(0)
            val x1 = tl.first
            val y1 = tl.second
            val br = results.rect.get(1)
            val x2 = br.first
            val y2 = br.second
            val w = x2- x1
            val h = y2- y1
            val circle_rad = min(w,h)/STROKE_RATIO
            results.output_iris.forEachIndexed{ iIrisp, pos ->
                originalSizeCanvas.drawCircle(
                    pos.first + x1,
                    pos.second + y1,
                    circle_rad,
                    paintCircle
                )
            }
            return output
        }
    }

}
