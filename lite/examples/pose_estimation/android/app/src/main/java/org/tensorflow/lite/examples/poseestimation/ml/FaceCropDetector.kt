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
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.Utils
import org.tensorflow.lite.examples.poseestimation.data.*
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.lang.Float.min
import java.lang.Float.max

class FaceCropDetector(
    private val interpreter: Interpreter,
    private val interopt: Interpreter.Options): AbstractDetector<List<FaceCrop>>() {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val scoreThreshold = 0.7f
        private const val nmsThreshold = 0.3f
        private const val TAG = "FaceCrop"
//        https://google.github.io/mediapipe/solutions/models.html#face-detection
//        private const val MODEL_FILENAME = "face_detection_full_range.tflite"
        private const val MODEL_FILENAME = "face_detection_back.tflite"

        fun create(context: Context, device: Device): FaceCropDetector {
            val options = AbstractDetector.getOption(device, context)

            return FaceCropDetector(
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
    override var inference_results: List<FaceCrop> = listOf()
    private var lastInferenceTimeNanos: Long = -1
    private val inputWidth = interpreter.getInputTensor(0).shape()[1]
    private val inputHeight = interpreter.getInputTensor(0).shape()[2]
    private var cropHeight = 0f
    private var cropWidth = 0f
    private var cropSize = 0
    private val reversedSigmoid by lazy<Float> {
        Utils.reverseSigmoid(scoreThreshold.toDouble()).toFloat()
    }
    private val visualizationUtils:VisualizationUtils = VisualizationUtils()
    private lateinit var ssdAnchor:List<Pair<Float,Float>>
    init{
        ssdAnchor = SSDAnchors.ssd_generate_anchors(SSDAnchors.SSD_OPTIONS_BACK)
    }
    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): List<FaceCrop> {
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

        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        printInferenceTime(TAG)

        val anchors_buffer = outputMap[0] as Array<Array<FloatArray>>
        val faceconfidence_buffer = outputMap[1] as Array<Array<FloatArray>>

        val postProcessingStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val facecrops = postProcessModelOuputs(bitmap, anchors_buffer,
            interpreter.getOutputTensor(0).shape(),
            faceconfidence_buffer)
        val prine_facecrops = NMS(facecrops, scoreThreshold, nmsThreshold)

//        Log.i(
//            TAG,
//            String.format(
//                "Postprocessing took %.2f ms",
//                (SystemClock.elapsedRealtimeNanos() - postProcessingStartTimeNanos) / 1_000_000f
//            )
//        )

        return prine_facecrops
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

        // type: float32[1,2304,1]
        for (i in 0 until faceconfidence_buffer[0].size){

            val confidence = Utils.sigmoid(faceconfidence_buffer[0][i][0])
            // skip confidence < scoreThreshold
            if(confidence < scoreThreshold)
                continue
            for (j in 0 until new_boxes[i].size){
                new_boxes[i][j] = Pair(new_boxes[i][j].first*bitmap.width, new_boxes[i][j].second*bitmap.height)
            }
            faceCrops.add(
                FaceCrop(data = new_boxes[i].toList(), score = confidence)
            )
        }
        faceCrops.sortByDescending { it.score }

        return faceCrops.toList()
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
    override fun drawResultOnBitmap(bitmap: Bitmap): Bitmap {

        val outputBitmap = visualizationUtils.drawKeypoints(bitmap,getResults())
//        if (results.size > 0) {
//            Log.e("bbbb", results.size.toString())
//        }
        return outputBitmap
    }

    private fun overlap_similarity(box1: Pair<Pair<Float,Float>,Pair<Float,Float>>,box2: Pair<Pair<Float,Float>,Pair<Float,Float>>) : Float{
        val box1_tl = box1.first
        val box2_tl = box2.first
        val tl_x = max(box1_tl.first, box2_tl.first)
        val tl_y = max(box1_tl.second, box2_tl.second)

        val box1_br = box2.second
        val box2_br = box2.second
        val br_x = min(box1_br.first, box2_br.first)
        val br_y = min(box1_br.second, box2_br.second)
        if (tl_x > br_x || tl_y > br_y){
            return 0f
        }
        val area1 = (box1_br.first - box1_tl.first)*(box1_br.second - box1_tl.second)
        val area2 = (box2_br.first - box2_tl.first)*(box2_br.second - box2_tl.second)
        val area3 = (br_x - tl_x) * (br_y - tl_y)
        // Log.e("xxxx", "area1: " + area1+",area2: " + area2 + ",area3: " + area3)
        val denominator = area1+area2-area3
        if (denominator > 0)
            return area3/denominator
        else
            return 0f
    }
    private fun NMS(facecrops: List<FaceCrop>, scoreThreshold: Float, nmsThreshold:Float): List<FaceCrop>{
        """Return only most significant detections"""
        val kept_faces = mutableListOf<FaceCrop>()

        for (face in facecrops){
            if (face.score < scoreThreshold) // the face have been sorted by score
                break   // break because the remained are low score.
            var suppressed = false
            for (kept in kept_faces){
                val similarity = overlap_similarity(Pair(kept.data.get(0),kept.data.get(1)), Pair(face.data.get(0),face.data.get(1)))
                // Log.e("ssssss", "similarity " + similarity)

                if (similarity > nmsThreshold){
                    // too similir, kill!
                    suppressed = true
                    break
                }
            }
            if (!suppressed){
                kept_faces.add(face)
            }
        }
        return kept_faces.toList()
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
                originalSizeCanvas.drawText(
                    "    " + facecrop.score.toString(),
                    facecrop.data.get(0).first,
                    facecrop.data.get(0).second,
                    paintText
                )
            }
            return output
        }
    }
}
