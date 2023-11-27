#!/usr/bin/env python
# author: Matias Mattamala

import math

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from cv_bridge import CvBridge
from std_msgs.msg import Header

import numpy as np
import torch
import torch.nn.functional as F

import cv2

VIS_DISPARITY = 256
FX = 1075.0
FY = 1220.0


class RosInterface:
    def __init__(self):
        self.read_params()
        self.setup_ros()
        self.setup_publishers()

    def read_params(self):
        """Read parameters from parameter server"""

        # Subscription topics
        self._img_l_topic = rospy.get_param(
            "~img_l_topic", "/alphasense_driver_ros/cam0/color_rect/image/compressed"
        )
        # self._info_l_topic = rospy.get_param(
        #     "~info_l_topic", "/alphasense_driver_ros/cam0/color_rect/camera_info"
        # )
        self._img_r_topic = rospy.get_param(
            "~img_r_topic", "/alphasense_driver_ros/cam1/color_rect/image/compressed"
        )
        # self._info_r_topic = rospy.get_param(
        #     "~info_r_topic", "/alphasense_driver_ros/cam1/color_rect/camera_info"
        # )

        # Other parameters
        self._model_file = rospy.get_param("~model_file", "/root/mmstereo/model.pt")

        print("Finished Reading Params")

        # Load model
        self._model = torch.jit.load(self._model_file)
        print("Loaded Model")

    def setup_ros(self):
        print("Setting up ROS")
        self._bridge = CvBridge()
        self._img_l_sub = message_filters.Subscriber(self._img_l_topic, CompressedImage)
        # self._info_l_sub = message_filters.Subscriber(self._l_info_topic, CameraInfo)
        self._img_r_sub = message_filters.Subscriber(self._img_r_topic, CompressedImage)
        # self._info_r_sub = message_filters.Subscriber(self._r_info_topic, CameraInfo)

        self._time_sync = message_filters.ApproximateTimeSynchronizer(
            [self._img_l_sub, self._img_r_sub],
            10,
            slop=0.1,
        )
        self._time_sync.registerCallback(self.image_callback)
    
    def setup_publishers(self):  # Add this function
        # Publisher for the disparity image
        self._disparity_pub = rospy.Publisher("/disparity_image", Image, queue_size=10)

    def image_callback(self, img_l_msg, img_r_msg):
        # Convert to numpy
        np_img_l = self._bridge.compressed_imgmsg_to_cv2(img_l_msg)
        np_img_r = self._bridge.compressed_imgmsg_to_cv2(img_r_msg)
        height, width, _ = np_img_l.shape

        # Convert inputs from Numpy arrays scaled 0 to 255 to PyTorch tensors scaled from 0 to 1.
        left_tensor = np_img_l.astype(np.float32).transpose((2, 0, 1)) / 255.0
        right_tensor = np_img_r.astype(np.float32).transpose((2, 0, 1)) / 255.0
        left_tensor = torch.from_numpy(left_tensor).unsqueeze(0)
        right_tensor = torch.from_numpy(right_tensor).unsqueeze(0)

        # Crop inputs such that they don't need any padding when passing to the network.5)
        target_height = int(math.ceil(height / 16) * 16)
        target_width = int(math.ceil(width / 16) * 16)
        padding_x = target_width - width
        padding_y = target_height - height
        left_tensor = F.pad(left_tensor, (0, padding_x, 0, padding_y))
        right_tensor = F.pad(right_tensor, (0, padding_x, 0, padding_y))

        # Move model and inputs to GPU.
        self._model.cuda()
        self._model.eval()
        left_tensor = left_tensor.cuda()
        right_tensor = right_tensor.cuda()

        # Do forward pass on model and get output.
        with torch.no_grad():
            output, all_outputs = self._model(left_tensor, right_tensor)
        disparity = output["disparity"]
        disparity_small = output["disparity_small"]
        matchability = output.get("matchability", None)

        print(disparity.shape)
        print(disparity_small.shape)
        print(matchability.shape)

        print(disparity.type())

        cpu_float_array = disparity.cpu().numpy()
        cpu_float_array = cpu_float_array.squeeze()
        header = Header(stamp=rospy.Time.now())
        disparity_msg = self._bridge.cv2_to_imgmsg(cpu_float_array)
        # print(disparity_msg.encoding)
        disparity_msg.header = header
        disparity_msg.encoding = "32FC1"
        self._disparity_pub.publish(disparity_msg)

        print("Published Image")

        # cv2.imshow("vis", cpu_float_array)
        # cv2.waitKey(0)

        # scale = disparity.shape[3] // disparity_small.shape[3]

        # # Generate visualizations for network output.
        # disparity_vis = visualization.make_cv_disparity_image(
        #     disparity[0, 0, :, :], VIS_DISPARITY
        # )
        # disparity_small_vis = visualization.make_cv_disparity_image(
        #     disparity_small[0, 0, :, :], VIS_DISPARITY // scale
        # )
        # disparity_small_vis = cv2.resize(
        #     disparity_small_vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        # )
        # confidence_vis = visualization.make_cv_confidence_image(
        #     torch.exp(matchability[0, 0, :, :])
        # )
        # confidence_vis = cv2.resize(
        #     confidence_vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        # )

        # disparity_vis = disparity_vis[:height, :width, :]
        # disparity_small_vis = disparity_small_vis[:height, :width, :]
        # confidence_vis = confidence_vis[:height, :width, :]

        # # Put all the visualizations together and display in a window.
        # vis_top = cv2.hconcat([left, disparity_vis])
        # vis_bottom = cv2.hconcat([confidence_vis, disparity_small_vis])
        # vis = cv2.vconcat([vis_top, vis_bottom])

        # cv2.imshow("vis", vis)
        # cv2.waitKey(0)
