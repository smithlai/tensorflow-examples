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
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.data.*
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import kotlin.math.exp

class DashMLDetector(val detector_list: List<AbstractDetector<*>>): AbstractDetector<DashML> {

    companion object {
        private const val TAG = "DashML"
        fun create(context: Context): DashMLDetector {
            val detectors = listOf(
                MoveNet.create(context, Device.CPU,ModelType.Lightning),
                MobilenetDetector.create(context, Device.CPU),
                FaceMeshDetector.create(context, Device.CPU)
            )

            return DashMLDetector(detectors)
        }
    }

    private var lastInferenceTimeNanos: Long = -1

    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): DashML {

        var object_list : List<DetectedObject>? = null
        var person_list : List<Person>? = null
        var face_list : List<FaceMesh>? = null

        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        runBlocking {
            val mutableList = mutableListOf<Deferred<*>>()
            val job = launch {
                for (detector in detector_list) {
                    when (detector) {
                        is EfficientDetector -> {
                            mutableList.add(async {
//                                Thread.sleep(300L)
                                object_list = detector?.inferenceImage(bitmap)
                                Log.e("aaaaxxxx", object_list?.size.toString())
                            })
                        }
                        is ObjectDetector -> {
                            mutableList.add(async {
//                                Thread.sleep(300L)
                                object_list = detector?.inferenceImage(bitmap)
                                Log.e("aaaaxxxx", object_list?.size.toString())
                            })
                        }
                        is PoseDetector -> {
                            mutableList.add(async {
//                                Thread.sleep(300L)
                                person_list = detector?.inferenceImage(bitmap)
                            })
                        }
                        is FaceMeshDetector -> {
                            mutableList.add(async {
//                                Thread.sleep(300L)
                                face_list = detector?.inferenceImage(bitmap)
                            })
                        }
                        else -> {

                        }
                    }
                }
                mutableList.awaitAll()
            }
            job.invokeOnCompletion {
                println("Completed, duration: ${System.currentTimeMillis() - inferenceStartTimeNanos} ms.")
            }
        }

//        for(detector in detector_list){
//            when(detector) {
//                is EfficientDetector -> {
//                    object_list = detector.inferenceImage(bitmap)
//                }
//                is PoseDetector -> {
//                    person_list = detector.inferenceImage(bitmap)
//                }
//                is FaceMeshDetector -> {
//                    face_list = detector.inferenceImage(bitmap)
//                }
//                else ->{
//
//                }
//            }
//        }
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        Log.e(
            TAG,
            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
        )
        return DashML(person_list, object_list, face_list)
    }


    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos
    override fun close() {
        for(detector in detector_list){
            detector?.close()
        }
    }

    override fun drawKeypoints(bitmap: Bitmap, results: DashML ): Bitmap {
        var tmpbmp = bitmap
        for(detector in detector_list) {
            when (detector) {
                is EfficientDetector -> {
                    detector.let { _detector ->
                        results.object_list?.let { result_list ->
                            tmpbmp=_detector.drawKeypoints(tmpbmp, result_list)
                        }
                    }
                }
                is ObjectDetector -> {
                    detector.let { _detector ->
                        results.object_list?.let { result_list ->
                            tmpbmp=_detector.drawKeypoints(tmpbmp, result_list)
                        }
                    }
                }
                is PoseDetector -> {
                    detector.let { _detector ->
                        results.pose_list?.let { result_list ->
                            tmpbmp=_detector.drawKeypoints(tmpbmp, result_list)
                        }
                    }
                }
                is FaceMeshDetector -> {
                    detector.let { _detector ->
                        results.face_list?.let { result_list ->
                            tmpbmp=_detector.drawKeypoints(tmpbmp, result_list)
                        }

                    }
                }
                else -> {

                }
            }
        }
        return tmpbmp
    }

}