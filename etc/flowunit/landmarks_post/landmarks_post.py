# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import _flowunit as modelbox
import json
import numpy as np


class landmarks_postFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        # 定义功能单元需要的配置参数
        self.kps = config.get_int('kps', 15)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # 从DataContext中获取输入输出BufferList对象
        in_feat = data_context.input("in_feat")
        out_data = data_context.output("out_data")

        # 循环处理每一个输入Buffer数据
        for buffer_feat in in_feat:
            # 将输入Buffer转换为Python对象
            feat_data = np.array(buffer_feat.as_object(), copy=False)
            
            # 业务处理：解码关键点模型的输出数据，得到猪只关键点坐标，转化为json数据
            feat_data = feat_data.reshape((self.kps, 2))
            feat_data = feat_data.tolist()
            
            result = {"landmarks_result": str(feat_data)}            
            result_str = json.dumps(result)
            
            # 将业务处理返回的结果数据转换为Buffer
            out_buffer = modelbox.Buffer(self.get_bind_device(), result_str)

            # 将输出Buffer放入输出BufferList中
            out_data.push_back(out_buffer)

        # 返回成功标志，ModelBox框架会将数据发送到后续的功能单元
        return modelbox.Status.StatusCode.STATUS_SUCCESS        

    def close(self):
        return modelbox.Status()
