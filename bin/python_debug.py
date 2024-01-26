#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.


import os
import sys
import datetime
import platform
import glob
import time
import pathlib


def get_sdk_path(base_path: str) -> str:
    for filename in os.listdir(base_path):
        file_path = os.path.join(base_path, filename)
        if (os.path.isdir(file_path) and filename.startswith("modelbox")):
            return file_path
    print("can not find sdk, exit")
    sys.exit(1)


__pythoncmd = 'python3'
__curpath = os.path.split(os.path.realpath(__file__))[0]
__sdk_path = get_sdk_path(os.path.dirname(
    os.path.dirname(os.path.dirname(__curpath))))
__python_install_path = os.path.join(__sdk_path, "python-install")
if not os.path.exists(__python_install_path):
    os.makedirs(__python_install_path)
sys.path.append(__python_install_path)

if platform.system().lower() == 'windows':
    __pythoncmd = 'python.exe'
    os.add_dll_directory(os.path.join(__sdk_path, "bin"))
try:
    import modelbox
except Exception as e:
    print(e)
    whl_path = glob.glob(__sdk_path + '/*.whl')[0]
    os.system(__pythoncmd + ' -m pip install pydevd --target=' +
              __python_install_path + ' --timeout=3 --retries=1')
    os.system(__pythoncmd + ' -m pip install ' + whl_path + ' --target=' + __python_install_path +
              ' --timeout=3 --retries=1')
    os.system(__pythoncmd + ' -m pip install toml --target=' +
              __python_install_path + ' --timeout=3 --retries=1')
    print('\n\nauto install modelbox and pydevd finish, please re-run it')
    sys.exit(0)

try:
    import toml
except Exception as e:
    print(e)
    os.system(__pythoncmd + ' -m pip install toml --target=' +
              __python_install_path + ' --timeout=3 --retries=1')
    print('\n\nauto install toml finish, please re-run it')
    sys.exit(0)


__log = modelbox.Log()


def log_level_str_to_level(level: str) -> modelbox.Log.Level:
    '''
    log level str to enum level
    '''
    if level == 'DEBUG':
        return modelbox.Log.Level.DEBUG
    elif level == 'INFO':
        return modelbox.Log.Level.INFO
    elif level == 'NOTICE':
        return modelbox.Log.Level.NOTICE
    elif level == 'WARN':
        return modelbox.Log.Level.WARN
    elif level == 'ERROR':
        return modelbox.Log.Level.ERROR
    elif level == 'FATAL':
        return modelbox.Log.Level.FATAL

    return modelbox.Log.Level.OFF


def log_level_to_str(level: modelbox.Log.Level) -> str:
    '''
    log level str to enum level
    '''
    if level == modelbox.Log.Level.DEBUG:
        return 'DEBUG'
    elif level == modelbox.Log.Level.INFO:
        return 'INFO'
    elif level == modelbox.Log.Level.NOTICE:
        return 'NOTICE'
    elif level == modelbox.Log.Level.WARN:
        return 'WARN'
    elif level == modelbox.Log.Level.ERROR:
        return 'ERROR'
    elif level == modelbox.Log.Level.FATAL:
        return 'FATAL'

    return "NONE"


def log_callback(level: modelbox.Log.Level, file: str, lineno: int, func: str, msg: str):
    '''
    Log callback function
    '''
    print("[{time}][{level:>5}][{file:>20}:{lineno:>4}] {msg}".format(
        time=datetime.datetime.now(), level=log_level_to_str(level),
        file=file, lineno=lineno, msg=msg
    ))


def reg_log():
    '''
    Register log function
    '''
    __log.reg(log_callback)
    __log.set_log_level(modelbox.Log.Level.INFO)


def set_log_level(level: modelbox.Log.Level):
    '''
    Set log level
    '''
    __log.set_log_level(level)


def change_define(url: str) -> str:
    url = url.replace('${HILENS_APP_ROOT}', os.path.dirname(
        __curpath).replace('\\', '/'))
    url = url.replace('${HILENS_DATA_DIR}', (os.path.dirname(
        __curpath) + '/hilens_data_dir').replace('\\', '/'))
    return url


def get_content(mock_cfg_common):
    if 'content_file' in mock_cfg_common.keys():
        with open(mock_cfg_common[content_file], 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return mock_cfg_common['content']
    return "{}"


def send_external_data(flow, mock_cfg, last_out, id):
    extern_data = flow.create_external_data_map()
    # 申请Buffer
    buffer_list = extern_data.create_buffer_list()
    mock_cfg['input' +
             id]['url'] = change_define(mock_cfg['input' + id]['url'])

    input_type = 'file'
    if (mock_cfg['input' + id]['type'] == 'rtsp'):
        input_type = 'stream'
    result_str = '{"url":"' + \
        mock_cfg['input' + id]['url'] + '","url_type":"' + input_type + '"}'

    sess_cfg = extern_data.get_session_config()
    sess_cfg.set("nodes.iva_task_output",
                 '{"data":{"url" : "' + last_out + '"}}')
    sess_cfg.set("nodes.iva_task_common", get_content(mock_cfg['common']))
    buffer_list.push_back(result_str)
    buffer_list[0].set('source_type', 'url')

    # 将数据发送到"input"
    extern_data.send("input1", buffer_list)
    # 结束输入
    extern_data.close()


def run_flow(flow_file: str):
    '''
    Run modelbox flow
    '''
    try:
        with open(flow_file) as f:
            pass
    except IOError as e:
        print("Open file {} failed, {}".format(
            flow_file, os.strerror(e.errno)), file=sys.stderr)
        return 1

    if (not os.path.exists(__curpath + "/mock_task.toml")):
        print("mock_task.toml dose not exist!")
        return 1

    graph_ext = os.getenv('HILENS_MB_GRAPH_TYPE', 'null')
    if graph_ext == 'null':
        mock_cfg = toml.load(__curpath + "/mock_task.toml")
    else:
        mock_cfg = toml.load(__curpath + "/mock_task" + graph_ext + ".toml")

    # initialize flow
    flow = modelbox.Flow()
    ret = flow.init(flow_file)
    if ret == False:
        modelbox.error(ret)
        return 1

    # build graph
    ret = flow.build()
    if ret == False:
        modelbox.error(ret)
        return 1

    # run flow async
    ret = flow.run_async()
    if ret == False:
        modelbox.error(ret)
        return 1

    last_out = mock_cfg['output']['url'] = change_define(
        mock_cfg['output']['url'])
    send_external_data(flow, mock_cfg, last_out, "")
    for i in range(1, 16):
        idstr = str(i)
        if ('input' + idstr not in mock_cfg.keys()):
            break
        if ('output' + idstr in mock_cfg.keys()):
            last_out = mock_cfg['output' + idstr]['url'] = change_define(
                mock_cfg['output' + idstr]['url'])
        time.sleep(0.1)
        send_external_data(flow, mock_cfg, last_out, idstr)

    # get flow result
    retval = modelbox.Status()
    ret = flow.wait(0, retval)
    if ret == False:
        modelbox.error(ret)
        return 1
    modelbox.info("Running result: " + str(retval))
    return 0


if __name__ == '__main__':
    arg_len = len(sys.argv)
    if (arg_len < 2):
        print("need flow path")
        sys.exit(1)

    pathlib.Path(__curpath+"/../hilens_data_dir").mkdir(parents=True, exist_ok=True)
    reg_log()
    set_log_level(log_level_str_to_level('INFO'))

    ret = run_flow(str(sys.argv[1]))
    sys.exit(ret)
