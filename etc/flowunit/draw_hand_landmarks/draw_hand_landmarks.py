# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _flowunit as modelbox
import cv2
import json
import numpy as np


class draw_hand_landmarksFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        # 定义功能单元需要的配置参数
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # 从DataContext中获取输入输出BufferList对象
        in_image = data_context.input("in_image")
        in_landmarks = data_context.input("in_landmarks")
        out_image = data_context.output("out_image")

        # 循环处理每一个输入Buffer数据
        for buffer_img, buffer_landmarks in zip(in_image, in_landmarks):
            # 获取输入图像Buffer的宽、高、通道数等属性信息
            width = buffer_img.get('width')
            height = buffer_img.get('height')
            channel = buffer_img.get('channel')
            
            # 将输入Buffer转换为numpy对象
            img_data = np.array(buffer_img.as_object(), dtype=np.uint8, copy=False)
            img_data = img_data.reshape(height, width, channel)

            max_bbox = buffer_img.get("bbox")
            label = buffer_img.get('label')
            
            # 将输入Buffer转换为字符串，从json字符串中解码出猪只关键点数据
            landmarks_str = buffer_landmarks.as_object()
            landmarks = self.decode_landmarks(landmarks_str)

            # 业务处理：将最大的猪只检测框对应的猪只关键点数据画在图上
            img_out = img_data.copy()
            if len(landmarks):
                landmarks = np.array(landmarks)
                self.draw_landmarks(img_out, landmarks, max_bbox, label)            
            
            # 将业务处理返回的结果数据转换为Buffer
            out_buffer = modelbox.Buffer(self.get_bind_device(), img_out)

            # 设置输出Buffer的Meta信息，此处直接拷贝输入Buffer的Meta信息
            out_buffer.copy_meta(buffer_img)

            # 将输出Buffer放入输出BufferList中
            out_image.push_back(out_buffer)

        # 返回成功标志，ModelBox框架会将数据发送到后续的功能单元
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def decode_landmarks(self, landmarks_str):
        """从json字符串中解码出猪只关键点数据"""
        try:
            landmarks_data = json.loads(landmarks_str)
            landmarks_list = json.loads(landmarks_data['landmarks_result'])
        except Exception as ex:
            modelbox.error(str(ex))
            return []
        else:
            return landmarks_list
    
    def draw_landmarks(self, out_img, landmarks, bbox, label):
        """将检测框和关键点画在图上"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out_img, f"{'pig'} {label[0]*100:.1f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 讲归一化后的关键点坐标还原到原图中的位置
        for x, y in landmarks:
            x = int(x * (x2 - x1) + x1) 
            y = int(y * (y2 - y1) + y1)
            cv2.circle(out_img, (x, y), 4, (255, 0, 0), -1)
