package org.tensorflow.lite.examples.poseestimation.data

data class DashML(val pose_list: List<Person>?, val object_list:List<DetectedObject>?, val face_list: List<FaceMesh>?)