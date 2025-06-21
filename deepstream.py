import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from ctypes import *

from utils import GETFPS, set_custom_bbox, parse_face_from_meta, bus_call, is_aarch64, rect_params_to_xyxy

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds
import numpy as np
MAX_ELEMENTS_IN_DISPLAY_META = 16

SOURCE = ''
CONFIG_PGIE_INFER = 'engine/primary/yolov8n_face.txt'
CONFIG_SGIE_INFER = 'engine/secondary/webface.txt'
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5
CNT = 0

fps_streams = {}

def probe(pad, info, user_data):
    global CNT
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    # print('let go ')
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        current_index = frame_meta.source_id
        # print('current_index ',current_index)

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            print('='*50)
            for attr in dir(obj_meta):
                if not attr.startswith("__"):  # skip built-in attributes
                    try:
                        value = getattr(obj_meta, attr)
                        print(f"{attr}: {value}")
                    except Exception as e:
                        print(f"{attr}: <Error accessing attribute> ({e})")
            print('='*50)
            CNT += 1
            print('CNT ',CNT)
            user_meta = obj_meta.obj_user_meta_list
            landmarks_list = []
            while user_meta:
                user_meta_data = pyds.NvDsUserMeta.cast(user_meta.data)
                if user_meta_data.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta_data.user_meta_data)
                    
                    print('tensor ',tensor_meta.output_layers_info(0))
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    output = np.array(
                        [
                            pyds.get_detections(layer.buffer, i)
                            for i in range(512)
                        ]
                    )

                    print(output)
                try:
                    user_meta = user_meta.next
                except StopIteration:
                    break

                        
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        fps_streams['stream{0}'.format(current_index)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
def primary_probe(pad, info, user_data):
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    # print('let go ')
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        current_index = frame_meta.source_id
        # print('current_index ',current_index)

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            print('@'*50)
            for attr in dir(obj_meta):
                if not attr.startswith("__"):  # skip built-in attributes
                    try:
                        value = getattr(obj_meta, attr)
                        print(f"{attr}: {value}")
                    except Exception as e:
                        print(f"{attr}: <Error accessing attribute> ({e})")
            print('@'*50)
            user_meta = obj_meta.obj_user_meta_list
            landmarks_list = []
            while user_meta:
                user_meta_data = pyds.NvDsUserMeta.cast(user_meta.data)
                if user_meta_data.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta_data.user_meta_data)
                    
                    print('tensor ',tensor_meta.output_layers_info(0))
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    output = np.array(
                        [
                            pyds.get_detections(layer.buffer, i)
                            for i in range(512)
                        ]
                    )

                    print(output.shape)
                try:
                    user_meta = user_meta.next
                except StopIteration:
                    break

                        
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        fps_streams['stream{0}'.format(current_index)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find('decodebin') != -1:
        Object.connect('child-added', decodebin_child_added, user_data)
    if name.find('nvv4l2decoder') != -1:
        Object.set_property('drop-frame-interval', 0)
        Object.set_property('num-extra-surfaces', 1)
        if is_aarch64():
            Object.set_property('enable-max-performance', 1)
        else:
            Object.set_property('cudadec-memtype', 0)
            Object.set_property('gpu-id', GPU_ID)


def cb_newpad(decodebin, pad, user_data):
    streammux_sink_pad = user_data
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)
    if name.find('video') != -1:
        if features.contains('memory:NVMM'):
            if pad.link(streammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write('ERROR: Failed to link source to streammux sink pad\n')
        else:
            sys.stderr.write('ERROR: decodebin did not pick NVIDIA decoder plugin')


def create_uridecode_bin(stream_id, uri, streammux):
    bin_name = 'source-bin-%04d' % stream_id
    bin = Gst.ElementFactory.make('uridecodebin', bin_name)
    if 'rtsp://' in uri:
        pyds.configure_source_for_ntp_sync(bin)
    bin.set_property('uri', uri)
    pad_name = 'sink_%u' % stream_id
    streammux_sink_pad = streammux.get_request_pad(pad_name)
    bin.connect('pad-added', cb_newpad, streammux_sink_pad)
    bin.connect('child-added', decodebin_child_added, 0)
    fps_streams['stream{0}'.format(stream_id)] = GETFPS(stream_id)
    return bin

def probe_caps(pad, info, user_data):
    caps = pad.get_current_caps()
    if caps:
        print(f"[Pgie] Pad {pad.get_name()} caps: {caps.to_string()}")
    else:
        print(f"[Pgie] Pad {pad.get_name()} has no caps yet")
    return Gst.PadProbeReturn.OK

def run(SOURCE):
    Gst.init(None)

    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write('ERROR: Failed to create pipeline\n')
        sys.exit(1)

    streammux = Gst.ElementFactory.make('nvstreammux', 'nvstreammux')
    if not streammux:
        sys.stderr.write('ERROR: Failed to create nvstreammux\n')
        sys.exit(1)
        
    source_bin = create_uridecode_bin(0, SOURCE, streammux)
    if not source_bin:
        sys.stderr.write('ERROR: Failed to create source_bin\n')
        sys.exit(1)

    pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
    if not pgie:
        sys.stderr.write('ERROR: Failed to create pgie nvinfer\n')
        sys.exit(1)

    sgie = Gst.ElementFactory.make('nvinfer', 'sgie')
    if not sgie:
        sys.stderr.write('ERROR: Failed to create sgie nvinfer\n')
        sys.exit(1)

    converter = Gst.ElementFactory.make('nvvideoconvert', 'nvvideoconvert')
    if not converter:
        sys.stderr.write('ERROR: Failed to create nvvideoconvert\n')
        sys.exit(1)

    osd = Gst.ElementFactory.make('nvdsosd', 'nvdsosd')
    if not osd:
        sys.stderr.write('ERROR: Failed to create nvdsosd\n')
        sys.exit(1)


    sink = Gst.ElementFactory.make('fakesink', 'fakesink')
    if not sink:
        sys.stderr.write('ERROR: Failed to create fakesink\n')
        sys.exit(1)

    sys.stdout.write('\n')
    sys.stdout.write('SOURCE: %s\n' % SOURCE)
    sys.stdout.write('CONFIG_PGIE_INFER: %s\n' % CONFIG_PGIE_INFER)
    sys.stdout.write('CONFIG_SGIE_INFER: %s\n' % CONFIG_SGIE_INFER)
    sys.stdout.write('STREAMMUX_BATCH_SIZE: %d\n' % STREAMMUX_BATCH_SIZE)
    sys.stdout.write('STREAMMUX_WIDTH: %d\n' % STREAMMUX_WIDTH)
    sys.stdout.write('STREAMMUX_HEIGHT: %d\n' % STREAMMUX_HEIGHT)
    sys.stdout.write('GPU_ID: %d\n' % GPU_ID)
    sys.stdout.write('PERF_MEASUREMENT_INTERVAL_SEC: %d\n' % PERF_MEASUREMENT_INTERVAL_SEC)
    sys.stdout.write('JETSON: %s\n' % ('TRUE' if is_aarch64() else 'FALSE'))
    sys.stdout.write('\n')

    streammux.set_property('batch-size', STREAMMUX_BATCH_SIZE)
    streammux.set_property('batched-push-timeout', 25000)
    streammux.set_property('width', STREAMMUX_WIDTH)
    streammux.set_property('height', STREAMMUX_HEIGHT)
    streammux.set_property('enable-padding', 0)
    
    streammux.set_property('attach-sys-ts', 1)
    if 'file://' in SOURCE:
        streammux.set_property('live-source', 0)
    else:
        streammux.set_property('live-source', 1)
    pgie.set_property('config-file-path', CONFIG_PGIE_INFER)
    pgie.set_property('qos', 0)
    sgie.set_property('config-file-path', CONFIG_SGIE_INFER)
    sgie.set_property('qos', 0)
    converter.set_property('qos', 0)
    osd.set_property('process-mode', int(pyds.MODE_GPU))
    osd.set_property('qos', 0)
    sink.set_property('async', 0)
    sink.set_property('sync', 0)
    sink.set_property('qos', 0)

    if not is_aarch64():
        streammux.set_property('nvbuf-memory-type', 0)
        streammux.set_property('gpu_id', GPU_ID)
        pgie.set_property('gpu_id', GPU_ID)
        # sgie.set_property('gpu_id', GPU_ID)
        converter.set_property('nvbuf-memory-type', 0)
        converter.set_property('gpu_id', GPU_ID)
        osd.set_property('gpu_id', GPU_ID)
    
    # Add elements
    pipeline.add(source_bin)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(converter)
    pipeline.add(osd)
    pipeline.add(sink)

    # Link elements
    streammux.link(pgie)
    # pgie.link(converter)
    pgie.link(sgie)
    sgie.link(converter)
    converter.link(osd)
    osd.link(sink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    # pgie_src_pad = pgie.get_static_pad("src")
    # pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, probe_caps, None)
    print('pgie ')
    pgie_src_pad = pgie.get_static_pad('src')
    if not pgie_src_pad:
        sys.stderr.write('ERROR: Failed to get pgie src pad\n')
        sys.exit(1)
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, primary_probe, 0)

    #####
    print('second gie ')

    sgie_src_pad = sgie.get_static_pad('src')
    if not sgie_src_pad:
        sys.stderr.write('ERROR: Failed to get pgie src pad\n')
        sys.exit(1)
    sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, probe, 0)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

    sys.stdout.write('\n')


if __name__ == '__main__':
    run('file:///app/assets/videos/CR7.mp4')
