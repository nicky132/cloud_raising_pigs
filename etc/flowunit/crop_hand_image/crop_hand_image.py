# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _flowunit as modelbox
import numpy as np
import cv2


class crop_hand_imageFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        # 获取功能单元的配置参数
        self.pose_net_h = config.get_int('pose_net_h', 224)
        self.pose_net_w = config.get_int('pose_net_w', 224)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # 从DataContext中获取输入输出BufferList对象
        in_data = data_context.input("in_data")
        out_image = data_context.output("roi_image")

        # 循环处理每一个输入Buffer数据
        for buffer_img in in_data:
            # 获取输入Buffer的属性信息
            width = buffer_img.get('width')
            height = buffer_img.get('height')
            channel = buffer_img.get('channel')

            # 将输入Buffer转换为numpy对象
            img_data = np.array(buffer_img.as_object(), dtype=np.uint8, copy=False)
            img_data = img_data.reshape(height, width, channel)

            max_bbox = buffer_img.get("bbox")

            # 业务处理：找出最大的猪只检测框，裁剪出猪只图像
            x1, y1, x2, y2 = max_bbox
            img_roi = img_data[y1:y2, x1:x2, :].copy()

            img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            img_roi = cv2.resize(img_roi, (self.pose_net_h, self.pose_net_w))

            # 将裁剪出的猪只图像转换为Buffer
            img_buffer = modelbox.Buffer(self.get_bind_device(), img_roi)

            # 将输出Buffer放入输出BufferList中
            out_image.push_back(img_buffer)

        # 返回成功标志，ModelBox框架会将数据发送到后续的功能单元
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
