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
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.*
import org.tensorflow.lite.examples.poseestimation.data.*

class DashMLDetector(val detector_list: List<AbstractDetector<*>>): AbstractDetector<DashML>() {

    companion object {
        private const val TAG = "DashML"
        fun create(context: Context): DashMLDetector {
            val detectors = listOf(
//                ===== Light =====
                MoveNet.create(context, Device.CPU,ModelType.Lightning),
                MobilenetDetector.create(context, Device.CPU),
                MultiFaceMeshDetector.create(context, Device.CPU)

//                ===== Heavy =====
//                MoveNetMultiPose.create(context, Device.CPU, Type.Dynamic),
//                EfficientDetector.create(context, Device.CPU),
//                FaceMeshDetector.create(context, Device.CPU)

            )

            return DashMLDetector(detectors)
        }
    }
    override var inference_results: DashML = DashML(null, null, null)
    private var lastInferenceTimeNanos: Long = -1

    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): DashML {

        var object_list : List<DetectedObject>? = null
        var person_list : List<Person>? = null
        var face_list : List<FaceMesh>? = null

        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        runBlocking(Dispatchers.IO) {
            val mutableList = mutableListOf<Job>()
            for (detector in detector_list) {
                when (detector) {
//                        is EfficientDetector -> {
//                            mutableList.add(launch {
////                                Thread.sleep(300L)
//                                object_list = detector?.inferenceImage(bitmap)
////                                Log.e("aaaaxxxx", "1.${Thread.currentThread().name}"+": "+object_list?.size.toString())
//                            })
//                        }
                    is ObjectDetector -> {
                        mutableList.add(launch {
//                                Thread.sleep(300L)
                            object_list = detector?.requestInferenceImage(bitmap)
//                            Log.e("aaaaxxxx", "2.${Thread.currentThread().name}"+": "+object_list?.size.toString())
                        })
                    }
                    is PoseDetector -> {
                        mutableList.add(launch {
//                                Thread.sleep(300L)
                            person_list = detector?.requestInferenceImage(bitmap)
//                            Log.e("aaaaxxxx", "3.${Thread.currentThread().name}"+": "+person_list?.size.toString())
                        })
                    }
                    is MultiFaceMeshDetector -> {
                        mutableList.add(launch {
//                                Thread.sleep(300L)
                            face_list = detector?.requestInferenceImage(bitmap)
//                            Log.e("aaaaxxxx", "4.${Thread.currentThread().name}"+": "+face_list?.size.toString())
                        })
                    }
                    else -> {

                    }
                }
            }

        }
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        Log.i(
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
            detector.drawKeypoints(bitmap)
        }
        return tmpbmp
    }
}
