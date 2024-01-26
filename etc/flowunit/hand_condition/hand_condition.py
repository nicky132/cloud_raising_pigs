# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _flowunit as modelbox
import json
import numpy as np


class hand_conditionFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # 从DataContext中获取输入输出BufferList对象
        in_image = data_context.input("in_image")
        in_bbox = data_context.input("in_bbox")

        has_hand = data_context.output("has_hand")
        no_hand = data_context.output("no_hand")

        # 循环处理每一个输入Buffer数据（实际上条件功能单元的batch size为1，此处循环中只有1条数据）
        for buffer_img, buffer_bbox in zip(in_image, in_bbox):
            # 获取输入Buffer的属性信息
            width = buffer_img.get('width')
            height = buffer_img.get('height')
            channel = buffer_img.get('channel')

            # 将输入Buffer转换为numpy对象
            img_data = np.array(buffer_img.as_object(), copy=False)
            img_data = img_data.reshape((height, width, channel))

            # 字符串数据可以直接用as_object函数转换
            bbox_str = buffer_bbox.as_object()

            # 解码出猪只检测框数据
            hand_bboxes = self.decode_hand_bboxes(bbox_str)
            
            # 此处是将输入Buffer直接作为输出Buffer向后传递
            # 此时Buffer的Data、Meta等全部内容都将保留，无需构建Buffer、设置Meta
            if hand_bboxes:  # 检测到猪只时的输出分支
                max_roi, label = self.get_max_roi(hand_bboxes, img_data)
                buffer_img.set("bbox", max_roi) # 将检测猪只框作为属性附在输出Buffer上
                buffer_img.set("label", label)  # 将检测猪只类别和置信度作为属性附在输出Buffer上
                has_hand.push_back(buffer_img)
            else:            # 未检测到猪只时的输出分支
                no_hand.push_back(buffer_img)

        # 返回成功标志，ModelBox框架会将数据发送到后续的功能单元
        return modelbox.Status.StatusCode.STATUS_SUCCESS
    
    def decode_hand_bboxes(self, bbox_str):
        """从json字符串中解码出猪只检测框"""
        try:
            hand_labels = [0, 1]  # 猪只对应的类别号是 0, 1
            det_result = json.loads(bbox_str)['det_result']
            if det_result == "None":
                return []
            bboxes = json.loads(det_result)
            hand_bboxes = list(filter(lambda x: int(x[5]) in hand_labels, bboxes))
        except Exception as ex:
            modelbox.error(str(ex))
            return []
        else:
            return hand_bboxes
    
    def get_max_roi(self, bboxes, img_data):
        """找出roi最大的猪只检测框"""
        max_bbox = max(bboxes, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]) * x[4])

        # 原始检测框数据归一化到[0,1]，此处需还原到原图中的坐标
        img_h, img_w, _ = img_data.shape
        x1, y1, x2, y2, score, clss = max_bbox
        x1 = int(x1 * img_w)
        y1 = int(y1 * img_h)
        x2 = int(x2 * img_w)
        y2 = int(y2 * img_h)
        new_bbox = [x1, y1, x2, y2]
        label = [score, clss]

        return new_bbox, label

    def close(self):
        return modelbox.Status()
