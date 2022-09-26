"""
@ capture multiple cameras images to different folder
"""
import numpy as np
import os, cv2, time, torch, random, pytorchvideo, warnings, argparse, math
import queue
import threading
from flask import Flask,render_template,Response
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image, )
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort
from collections import deque
import queue

import os
import time
import requests
import subprocess

import cv2
import numpy as np
import multiprocessing as mp
maxsize=24
def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img


def ava_inference_transform(clip, boxes,
                            num_frames=32,  # if using slowfast_r50_detection, change this to 32, 4 for slow
                            crop_size=640,
                            data_mean=[0.45, 0.45, 0.45],
                            data_std=[0.225, 0.225, 0.225],
                            slow_fast_alpha=4,  # if using slowfast_r50_detection, change this to 4, None for slow
                            ):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes, )
    clip = normalize(clip,
                     np.array(data_mean, dtype=np.float32),
                     np.array(data_std, dtype=np.float32), )
    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip, 1,
                                          torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), roi_boxes


def plot_one_box(x, img, color=[100, 100, 100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)
    return img


def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs


def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map,AI_queue):
    img_num = len(yolo_preds.ims)
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if ((i>=int((img_num-5))) and (pred.shape[0])):
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    # ava_label = ''
                    continue
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                # if (int(cls) != 0):#just only detecting the person
                #     continue
                print("avalabel:***{}***".format(ava_label))
                text = '{} {} {}'.format(int(trackid), yolo_preds.names[int(cls)], ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box, im, color, text)

            AI_queue.put(im)
            time.sleep(0.01)
            AI_queue.get() if AI_queue.qsize() > 1 else time.sleep(0.000001)

        # print('save_size:',p_result.qsize())


def show_yolopreds(yolo_preds,AI_queue):
    img_num = len(yolo_preds.ims)
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # print("pred:",pred,pred.shape)
        if pred.shape[0]:
            # yolo_preds.print()  # or .show(), .save(), .crop(), .pandas(), etc.
            # print('----------------',yolo_preds.xyxy[0])  # img1 predictions (tensor)
            imageIndex=yolo_preds.pandas().xyxy[i]  # img1 predictions (pandas)
            if imageIndex.shape[0]:
                for box_num in range(int(imageIndex.shape[0])):
                    startX = int(imageIndex["xmin"][box_num])
                    startY = int(imageIndex["ymin"][box_num])
                    endX = int(imageIndex["xmax"][box_num])
                    endY = int(imageIndex["ymax"][box_num])
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    im=cv2.putText(im, imageIndex["name"][box_num],(startX, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
                    im=cv2.rectangle(im,(startX, startY), (endX, endY), (0, 255, 0), 2)

        if i < int((img_num - 5)):
            AI_queue.put(im)
            # time.sleep(0.01)
            AI_queue.get() if AI_queue.qsize() > 1 else None
            time.sleep(0.01)
        # print('save_size:',AI_queue.qsize())

def load_model(yolo_modeldir,conf=0.4,iou=0.4):

    yolo_model=torch.hub.load(yolo_modeldir,'yolov5l6', source='local',pretrained=True)

    yolo_model.conf = conf
    yolo_model.iou = iou
    yolo_model.max_det = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    print("cuda:",torch.cuda.is_available())
    # video_model = slowfast_r50_detection(True).eval().to(device)
    slowfast_model = slowfast_r50_detection(True).eval().to(device)
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/ava_action_list.pbtxt")

    return yolo_model,slowfast_model,deepsort_tracker,ava_labelnames


def yolo_slowfast_action(model,slowfast_model,deepsort_tracker,ava_labelnames,framelist,AI_queue,maxsize=maxsize):
    # print(len(framelist),framelist)
    # model.conf=0.4
    imsize=640
    yolo_preds = model(framelist,size=imsize)
    show_yolopreds(yolo_preds,AI_queue)
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    img_num = int(maxsize)
    yolo_preds.files = [f"img_{k}.jpg" for k in range(img_num)]

    deepsort_outputs = []
    for j in range(len(yolo_preds.pred)):

        temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                   yolo_preds.ims[j])
        if len(temp) == 0:
            temp = np.ones((0, 8))
        deepsort_outputs.append(temp.astype(np.float32))
    yolo_preds.pred = deepsort_outputs

    id_to_ava_labels = {}
    video_clips=torch.from_numpy(np.array(framelist))
    # video_clips.transpose(3,2).transpose(2,1).transpose(1,0)
    video_clips=video_clips.permute((3,0,1,2))
    if yolo_preds.pred[img_num // 2].shape[0]:
        inputs, inp_boxes, _ = ava_inference_transform(video_clips, yolo_preds.pred[img_num // 2][:, 0:4],crop_size=imsize)
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            time1=time.time()
            slowfaster_preds = slowfast_model(inputs, inp_boxes.to(device))
            slowfaster_preds = slowfaster_preds.cpu()
            time2=time.time()
            cost=time2-time1
            print("cost:_____",cost)
        for tid, avalabel in zip(yolo_preds.pred[img_num // 2][:, 5].tolist(),np.argmax(slowfaster_preds, axis=1).tolist()):
            id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]
    save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map,AI_queue)

def image_put(queue,camera_queue):
    # camera_ip='./demo/B6-8-23-10-46-t.mp4'
    camera_ip=camera_queue
    if len(camera_ip)==1:
        camera_ip=int(camera_ip)

    cap = cv2.VideoCapture(camera_ip)
    is_opened, frame = cap.read()
    print('cam open:',is_opened)

    while is_opened:
        is_opened, frame = cap.read()
        queue.put(frame)
        if AI_queue.qsize() >=maxsize:
            time.sleep(0.01)
            AI_queue.get()
        else:
            None

    cv2.VideoCapture(camera_ip).release()

def show_result(flaskresult_queue):
    while True:
        if (flaskresult_queue.full() != True):
            print("show_result.................")
            result_frame = flaskresult_queue.get()
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            print("display_invalid:",flaskresult_queue.empty())

def AI_process(queue,AI_queue):
    yolo_model, slowfast_model, deepsort_tracker, ava_labelnames = load_model(yolo5l6)
    while True:
        if(queue.qsize()== maxsize):
            # print('len_queue')
            frame_list=[queue.get() for _ in range(maxsize)]
            # print(frame_list)
            yolo_slowfast_action(yolo_model,slowfast_model, deepsort_tracker,  ava_labelnames,frame_list,AI_queue)
        else:
            time.sleep(2)
            print("11111111111111111111111")
def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

def image_collect(queue_list,flaskresult_queue):
    """show in single opencv-imshow window"""
    window_name = "caoncat_cam_display"
    # cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        imgs =[cv2.resize(q.get(),(720,720)) for q in queue_list]
        # imgs=cv2.resize(np.array(imgs[0]),(256,256))
        imgs = np.concatenate(imgs, axis=1)
        flaskresult_queue.put(imgs)
        flaskresult_queue.get() if flaskresult_queue.qsize() > 1 else None
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(show_result(flaskresult_queue), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo5l6 = r'C:\Users\Administrator\.cache\torch\hub\ultralytics_yolov5_master'

# yolo_model, slowfast_model, deepsort_tracker, ava_labelnames = load_model(yolo5l6)
# mp.set_start_method(method='spawn')
AI_queue=mp.Queue(maxsize=2)
flaskresult_queue = mp.Queue(maxsize=2)
def run_multi_camera(rtsp='rtmpCollection.txt'):
    rtsp_list=[]
    with open(rtsp,'r') as rtspfile:
        rtsps=rtspfile.readlines()
        for rtsp_num in range(len(rtsps)):
            rtsp_ip=rtsps[rtsp_num].strip()
            rtsp_list.append(rtsp_ip)
    print(rtsp_list)


    frame_queues = [mp.Queue(maxsize=maxsize) for _ in rtsp_list]
    AI_result_queue=[mp.Queue(maxsize=2) for _ in rtsp_list]
    processes = [mp.Process(target=image_collect, args=(AI_result_queue,flaskresult_queue))]

    for input_queue,AI_result,cam_ip in zip(frame_queues,AI_result_queue,rtsp_list):
        processes.append(mp.Process(target=image_put, args=(input_queue,cam_ip)))
        processes.append(mp.Process(target=AI_process, args=(input_queue,AI_result)))
        # processes.append(mp.Process(target=image_collect,args=(AI_result_queue,)))
    # processes.append(mp.Process(target=app.run(port=5555)))
    for process in processes:
        process.daemon = True
        process.start()
    processes.append(mp.Process(target=app.run()))
    processes[-1].daemon = True
    processes[-1].start()
    for process in processes:
        process.join()


def run():
    run_multi_camera(rtsp='rtmpCollection.txt')
if __name__=='__main__':
    run()