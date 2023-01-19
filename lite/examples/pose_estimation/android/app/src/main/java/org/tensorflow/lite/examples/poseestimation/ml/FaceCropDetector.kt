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
import org.tensorflow.lite.examples.poseestimation.data.SSDAnchors
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.lang.Math.log
import kotlin.math.exp

class FaceCropDetector(
    private val interpreter: Interpreter,
    private var gpuDelegate: GpuDelegate?): AbstractDetector<List<FaceCrop>> {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val scoreThreshold = 0.7f
        private const val iouThreshold = 0.3f
        private const val THRESHOLD = 0.3f
        private const val TAG = "FaceCrop"
//        https://google.github.io/mediapipe/solutions/models.html#face-detection
//        private const val MODEL_FILENAME = "face_detection_full_range.tflite"
        private const val MODEL_FILENAME = "face_detection_back.tflite"

        fun create(context: Context, device: Device): FaceCropDetector {
            val settings: Pair<Interpreter.Options, GpuDelegate?> = AbstractDetector.getOption(device)
            val options = settings.first
            var gpuDelegate = settings.second

            return FaceCropDetector(
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
    private val reversedSigmoid by lazy<Float> {
        reverseSigmoid(scoreThreshold.toDouble()).toFloat()
    }
    private val visualizationUtils:VisualizationUtils = VisualizationUtils()
    private lateinit var ssdAnchor:List<Pair<Float,Float>>
    init{
        ssdAnchor = SSDAnchors.ssd_generate_anchors(SSDAnchors.SSD_OPTIONS_BACK)
    }
    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): List<FaceCrop> {
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
        Log.i(
            TAG,
            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
        )

        val anchors_buffer = outputMap[0] as Array<Array<FloatArray>>
        val faceconfidence_buffer = outputMap[1] as Array<Array<FloatArray>>

        val postProcessingStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val facecrops = postProcessModelOuputs(bitmap, anchors_buffer,
            interpreter.getOutputTensor(0).shape(),
            faceconfidence_buffer)
        Log.i(
            TAG,
            String.format(
                "Postprocessing took %.2f ms",
                (SystemClock.elapsedRealtimeNanos() - postProcessingStartTimeNanos) / 1_000_000f
            )
        )

        return facecrops
    }

    /**
     * Convert heatmaps and offsets output of Posenet into a list of keypoints
     */
    private fun postProcessModelOuputs(
        bitmap: Bitmap,
        anchors_buffer: Array<Array<FloatArray>>,
        shape: IntArray,
        faceconfidence_buffer: Array<Array<FloatArray>>
    ): List<FaceCrop> {


        var faceCrops = mutableListOf<FaceCrop>()
        // type: float32[1,2304,1]



        val new_boxes = SSDAnchors.decode_boxes(inputHeight, anchors_buffer, shape, ssdAnchor)

        for (i in 0 until faceconfidence_buffer[0].size){

            // type: float32[1,2304,1]
            if(faceconfidence_buffer[0][i][0] < reversedSigmoid)
                continue
            val confidence = faceconfidence_buffer[0][i][0]
            for (j in 0 until new_boxes[i].size){
                new_boxes[i][j] = Pair(new_boxes[i][j].first*bitmap.width, new_boxes[i][j].second*bitmap.height)
            }
            faceCrops.add(
                FaceCrop( data = new_boxes[i].toList(), score = confidence)
            )
        }
        if (faceCrops.size > 0) {
            Log.e("aaaa", "Find " + faceCrops.size.toString() + " face")
        }
        return faceCrops.toList()
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

        // float32[1,2304,16]
        val coordinations_shape = interpreter.getOutputTensor(0).shape()
//        Log.e("aaaa", coordinations_shape.joinToString(",") )
        outputMap[0] = Array(coordinations_shape[0]) {

            Array(coordinations_shape[1]) {
                FloatArray(coordinations_shape[2])
            }

        }

//        // type: float32[1,2304,1]
        val confidence_shape = interpreter.getOutputTensor(1).shape()
//        Log.e("aaaa", confidence_shape.joinToString(",") )
        outputMap[1] = Array(confidence_shape[0]) {
            Array(confidence_shape[1]) {
                FloatArray(confidence_shape[2])
            }
        }
        return outputMap
    }
    override fun drawKeypoints(bitmap: Bitmap, results: List<FaceCrop> ): Bitmap {
        val outputBitmap = visualizationUtils.drawKeypoints(bitmap,results)
        if (results.size > 0) {
            Log.e("bbbb", results.size.toString())
        }
        return outputBitmap
    }
    /** Returns value within [0,1].   */
    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-x)))
    }
    //    https://stackoverflow.com/questions/10097891/inverse-logistic-function-reverse-sigmoid-function
    /** Reverse sigmoid   */
    private fun reverseSigmoid(x: Double): Double {
        return Math.log(scoreThreshold / (1.0 - scoreThreshold))
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
            results: List<FaceCrop>
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
            results.forEach { facecrop ->
                facecrop.data.forEach { keypoint ->
                    originalSizeCanvas.drawCircle(
                        keypoint.first,
                        keypoint.second,
                        CIRCLE_RADIUS,
                        paintCircle
                    )
                }
            }
            return output
        }
    }
}
