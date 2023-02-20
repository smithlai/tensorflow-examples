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

class FaceMeshDetector(
    private val interpreter: Interpreter,
    private val irisDetector: LeftIrisDetector?,
    private val interopt: Interpreter.Options): AbstractDetector<List<FaceMesh>>() {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val THRESHOLD = 0.90f
        private const val STROKE_RATIO = 120
        private const val EYE_MARGIN = 0.25f
        private const val TAG = "FaceMesh"
//        https://google.github.io/mediapipe/solutions/models.html#face-mesh
        private const val MODEL_FILENAME = "face_landmark.tflite"
        private const val MODEL2_FILENAME = "face_landmark_with_attention.tflite"


        fun create(context: Context, device: Device, iris: Boolean): FaceMeshDetector {
            val options = AbstractDetector.getOption(device, context)
            val attention = false
            return FaceMeshDetector(
                Interpreter(
                    FileUtil.loadMappedFile(
                        context,
                        if (attention) MODEL2_FILENAME else MODEL_FILENAME
                    ), options
                ),
                if (iris) LeftIrisDetector.create(context, device) else null,
                options
            )
        }
    }

    override var inference_results: List<FaceMesh> = listOf()
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

        val outputMap = initOutputMap(interpreter)

//        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
//        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
//        Log.e(
//            TAG,
//            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
//        )

        val postProcessingStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val facemeshes = postProcessModelOuputs(bitmap, outputMap)
//        Log.i(
//            TAG,
//            String.format(
//                "Postprocessing took %.2f ms",
//                (SystemClock.elapsedRealtimeNanos() - postProcessingStartTimeNanos) / 1_000_000f
//            )
//        )
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        printInferenceTime(TAG)
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

    private fun detectIrisMesh(bitmap: Bitmap, iris_detecter: LeftIrisDetector, facepoints: List<Triple<Float,Float,Float>>, isLeft:Boolean=true) : IrisMesh{

        var upper = FaceMesh.left.get("EyebrowUpper")
        var lower = FaceMesh.left.get("EyeLower2")
        if (!isLeft){
            upper = FaceMesh.right.get("EyebrowUpper")
            lower = FaceMesh.right.get("EyeLower2")
        }
        // getTopleft
        var x1 = -1f
        var y1 = -1f
        var x2 = -1f
        var y2 = -1f

        listOf(upper, lower)?.forEach { contour ->
            contour?.forEach { idx ->
                val face = facepoints.get(idx)
                if (x1 < 0 || face.first < x1 ){
                    x1 = face.first
                }
                if (y1 < 0 || face.second < y1 ){
                    y1 = face.second
                }
                if (x2 < 0 || face.first > x2 ){
                    x2 = face.first
                }
                if (y2 < 0 || face.second > y2 ){
                    y2 = face.second
                }
            }
        }


        val cx = (x1+x2)/2
        val cy = (y1+y2)/2
        var w = (x2-x1)
        var h = (y2-y1)
        var w_2 = w*(1+EYE_MARGIN)/2
        var h_2 = h*(1+EYE_MARGIN)/2
        x1 = cx - w_2
        y1 = cy - h_2
        x2 = cx + w_2
        y2 = cy + h_2

        if ( x1 > 0 && y1 > 0 && x2 > 0 && x2 > 0 &&
            x1 < bitmap.width && y1 < bitmap.height &&
            x2 < bitmap.width && y2 < bitmap.height &&
            x2 >  x1 && y2 > y1){
            val w = x2 - x1
            val h = y2 - y1

            val eye = if(isLeft) {
                Bitmap.createBitmap(bitmap, x1.toInt(), y1.toInt(), w.toInt(), h.toInt())
            }else{
                val matrix = Matrix().apply { postScale(-1f, 1f, w / 2f, h / 2f) }
                Bitmap.createBitmap(
                    bitmap,
                    x1.toInt(),
                    y1.toInt(),
                    w.toInt(),
                    h.toInt(),
                    matrix,
                    true
                )
            }
            var irismesh = iris_detecter.requestInferenceImage(eye)


            if (!isLeft){
                val output_iris2 = irismesh.output_iris.toMutableList()
                for (i in 0 until irismesh.output_iris.size) {
                    val pt = irismesh.output_iris[i]
                    output_iris2[i] = Triple(w - pt.first, pt.second, pt.third)
                }
                irismesh.output_iris = output_iris2.toList()

                val output_eyes_contours_and_brows2 = irismesh.output_eyes_contours_and_brows.toMutableList()
                for (i in 0 until irismesh.output_eyes_contours_and_brows.size) {
                    val pt = irismesh.output_eyes_contours_and_brows[i]
                    output_eyes_contours_and_brows2[i] = Triple(w - pt.first, pt.second, pt.third)
                }
                irismesh.output_eyes_contours_and_brows = output_eyes_contours_and_brows2.toList()

            }
            val new_crop_points = listOf(Pair(x1,y1), Pair(x2,y2))

            irismesh.rect = new_crop_points

            return irismesh
        }
        return IrisMesh()
    }
    /**
     * Convert heatmaps and offsets output of Posenet into a list of keypoints
     */
    private fun postProcessModelOuputs(
        bitmap: Bitmap,
        outputMap: HashMap<Int,*>
    ): List<FaceMesh> {
        var outputcount = 0
        // All landmarks
        val landmark_buffer = outputMap[outputcount] as Array<Array<Array<FloatArray>>>
        // 1 * 1 * 1 * 1404 = 468 * 3
        val pos_buffer_p = landmark_buffer[0][0][0]
        val keypointPositions = flatToDimScale(3, pos_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)

//  ----  Encountered unresolved custom op: Landmarks2TransformMatrix.----------
//        var lip = listOf<Triple<Float,Float,Float>>()
//        var lefteye = IrisMesh()
//        var righteye = IrisMesh()
//
//        // MODEL2_FILENAME, face mesh with attention
//        if (interpreter.outputTensorCount >= 7) {
//            //1:lip
//            outputcount += 1
//            // 1 * 1 * 1 * 160 contains lips 80*2
//            val lip_buffer = outputMap[outputcount] as Array<Array<Array<FloatArray>>>
//            val lip_buffer_p = lip_buffer[0][0][0]
//            val lipPositions = flatToDimScale(2, lip_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)
//            // 2:left eye
//            outputcount += 1
//            // 1 * 1 * 1 * 142 contains eye 71*2
//            val le_buffer = outputMap[outputcount] as Array<Array<Array<FloatArray>>>
//            val le_buffer_p = le_buffer[0][0][0]
//            val lePositions = flatToDimScale(2, le_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)
//            // 3: right eye
//            outputcount += 1
//            // 1 * 1 * 1 * 142 contains eye 71*2
//            val re_buffer = outputMap[outputcount] as Array<Array<Array<FloatArray>>>
//            val re_buffer_p = re_buffer[0][0][0]
//            val rePositions = flatToDimScale(2, re_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)
//
//            // 4: left pupil
//            outputcount += 1
//            // 1 * 1 * 1 * 10 contains pupil 5*2
//            val lp_buffer = outputMap[outputcount] as Array<Array<Array<FloatArray>>>
//            val lp_buffer_p = lp_buffer[0][0][0]
//            val lpPositions = flatToDimScale(2, lp_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)
//
//            // 5: right pupil
//            outputcount += 1
//            // 1 * 1 * 1 * 10 contains pupil 5*2
//            val rp_buffer = outputMap[outputcount] as Array<Array<Array<FloatArray>>>
//            val rp_buffer_p = rp_buffer[0][0][0]
//            val rpPositions = flatToDimScale(2, rp_buffer_p, inputWidth, inputHeight, bitmap.width, bitmap.height)
//
//            lip = lipPositions.toList()
//            lefteye = IrisMesh(output_eyes_contours_and_brows = lePositions.toList(),
//                output_iris = lpPositions.toList())
//            righteye = IrisMesh(output_eyes_contours_and_brows = rePositions.toList(),
//                output_iris = rpPositions.toList())
//        }
//  ----  Encountered unresolved custom op: Landmarks2TransformMatrix.----------


        // Score
        outputcount += 1
        val faceflag_buffer = outputMap[outputcount] as Array<Array<Array<FloatArray>>>
        val faceflag = faceflag_buffer[0][0][0][0]
        val confidence = Utils.sigmoid(faceflag)
        var facelist = mutableListOf<FaceMesh>()
        var leye = IrisMesh()
        var reye = IrisMesh()
        if (confidence>= THRESHOLD) {
            val face_crop = FaceCrop(
                listOf(Pair(0f, 0f), Pair(bitmap.width.toFloat(), bitmap.height.toFloat())),
                1.0f
            )

            irisDetector?.let {  iris_detecter ->
                leye = detectIrisMesh(bitmap, iris_detecter, keypointPositions.toList(), isLeft = true)
                reye = detectIrisMesh(bitmap, iris_detecter, keypointPositions.toList(), isLeft = false)
            }
            facelist.add(
                FaceMesh(
                    keypoints = keypointPositions.toList(),
                    leftIrisMesh = leye,
                    rightIrisMesh = reye,
                    face_crop,
                    confidence = confidence
                )
            )
        }
        return facelist.toList()
    }

    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos
    override fun close() {
        interpreter.close()
        interopt.delegates.forEach {
            it.close()
        }
//        gpuDelegate?.close()
//        gpuDelegate = null
        irisDetector?.close()
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
        // 1 * 1 * 1 * 1404 =>  contains 468 3D points
        val coordinations_shape = interpreter.getOutputTensor(outputcount).shape()

        outputMap[outputcount] = Array(coordinations_shape[0]) {
            Array(coordinations_shape[1]) {
                Array(coordinations_shape[2]) {
                    FloatArray(coordinations_shape[3])
                }
            }
        }
        if (interpreter.outputTensorCount >= 7) {   // MODEL2_FILENAME, face mesh with attention
            //1:lips, 2:left eye, 3 right eye, 4: left pupil, 5: right pupil
            for (i in 1 .. 5) {
                outputcount += 1
                // 1 * 1 * 1 * 160 contains lips 80*2
                val shape = interpreter.getOutputTensor(outputcount).shape()
                outputMap[outputcount] = Array(shape[0]) {
                    Array(shape[1]) {
                        Array(shape[2]) {
                            FloatArray(shape[3])
                        }
                    }
                }
            }
        }

        outputcount += 1
        // 1 * 1 * 1 * 1 contains scores
        val score_shape = interpreter.getOutputTensor(outputcount).shape()
        outputMap[outputcount] = Array(score_shape[0]) {
            Array(score_shape[1]) {
                Array(score_shape[2]) {
                    FloatArray(score_shape[3])
                }
            }
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
                val tl = facemesh.facecrop.data.get(FaceCrop.FaceCropPart.TL.position)
                val x1 = tl.first
                val y1 = tl.second
                val br = facemesh.facecrop.data.get(FaceCrop.FaceCropPart.BR.position)
                val x2 = br.first
                val y2 = br.second
                val w = x2- x1
                val h = y2- y1
                val circle_rad = min(w,h)/STROKE_RATIO
                paintCircle.strokeWidth = circle_rad
                val eyes = listOf(
//                        listOf(FaceMesh.leftEyeUpper0, FaceMesh.leftEyeLower0,
//                                FaceMesh.rightEyeUpper0, FaceMesh.rightEyeLower0),
//                    listOf(FaceMesh.leftEyeUpper1, FaceMesh.leftEyeLower1,
//                        FaceMesh.rightEyeUpper1, FaceMesh.rightEyeLower1),
                    listOf(FaceMesh.left.get("EyebrowUpper"), FaceMesh.left.get("EyeLower2"),
                        /*FaceMesh.right.get("EyebrowUpper"), FaceMesh.right.get("EyeLower2")*/)
                )
                val colors = listOf(Color.BLUE, Color.RED, Color.GREEN)

                facemesh.leftIrisMesh.output_iris.forEachIndexed{ index, pt ->
                    originalSizeCanvas.drawCircle(
                        pt.first + x1 + facemesh.leftIrisMesh.rect.get(0).first,
                        pt.second + y1 + facemesh.leftIrisMesh.rect.get(0).second,
                        circle_rad,
                        paintCircle
                    )
                }
                facemesh.rightIrisMesh.output_iris.forEachIndexed{ index, pt ->
                    originalSizeCanvas.drawCircle(
                        pt.first + x1 + facemesh.rightIrisMesh.rect.get(0).first,
                        pt.second + y1 + facemesh.rightIrisMesh.rect.get(0).second,
                        circle_rad,
                        paintCircle
                    )
                }
                facemesh.leftIrisMesh.rect.forEachIndexed{ index, pt ->
                    originalSizeCanvas.drawCircle(
                        pt.first + x1,
                        pt.second + y1,
                        circle_rad,
                        paintCircle
                    )
                }
                facemesh.rightIrisMesh.rect.forEachIndexed{ index, pt ->
                    originalSizeCanvas.drawCircle(
                        pt.first + x1,
                        pt.second + y1,
                        circle_rad,
                        paintCircle
                    )
                }
                eyes.forEachIndexed { index1, eye_layer ->
                    paintCircle.color = colors.get(index1)
                    eye_layer.forEachIndexed { index2, eyelines ->
                        eyelines?.forEachIndexed { index3, eyepoints ->
                            val global_index = eyepoints
                            val pt = facemesh.keypoints[global_index]
                            originalSizeCanvas.drawCircle(
                                pt.first + x1,
                                pt.second + y1,
                                circle_rad,
                                paintCircle
                            )
                        }
                    }
                }
                paintCircle.color = Color.YELLOW
                FaceMesh.silhouette.forEach {
                    val pt = facemesh.keypoints[it]
                    originalSizeCanvas.drawCircle(
                        pt.first + x1,
                        pt.second + y1,
                        circle_rad,
                        paintCircle
                    )
                }

            }
            return output
        }
    }

}
