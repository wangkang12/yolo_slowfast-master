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
# q=queue.Queue(maxsize=25)
maxsize=24
q=deque(maxlen=maxsize)
p_result=queue.Queue(maxsize=2)
lock=threading.Lock()

saveimg=True
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


def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map,saveimg=False):
    img_num = len(yolo_preds.ims)
    save_img_actionlabel = ["bend/bow", "getup/squat", "carry/hold", "climb","smoke"]
    save_img_label=['backpack','handbag','suitcase']
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        yolo_label=''
        if ((i>=int((img_num-5))) and (pred.shape[0])):
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    yolo_label = yolo_preds.names[int(cls)]
                    ava_label=''
                    trackid=''
                    # continue
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                # if (int(cls) != 0):#just only detecting the person
                #     continue
                print("avalabel:***{}***".format(ava_label))
                # text = '{} {} {}'.format(str(trackid), yolo_preds.names[int(cls)], ava_label)
                text = '{} {}'.format(str(trackid), ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box, im, color, text)
                if saveimg:
                    if (ava_label in save_img_actionlabel)or(yolo_label in save_img_label):
                        imgname = time.ctime()
                        imgname = imgname.replace(' ', '_').replace(':', '_')
                        cv2.imwrite(todaya_time_folder + '//'+str(imgname) + '.jpg', im)
            p_result.put(im)
            time.sleep(0.05)
            p_result.get() if p_result.qsize() > 1 else time.sleep(0.000001)

        # print('save_size:',p_result.qsize())


def show_yolopreds(yolo_preds):
    img_num = len(yolo_preds.ims)
    save_img_label=['person','backpack','handbag','suitcase']
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):

        if pred.shape[0]:
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
            p_result.put(im)
            # time.sleep(0.01)
            p_result.get() if p_result.qsize() > 1 else time.sleep(0.000001)
        #
        # print('save_size:',p_result.qsize())

def load_model(yolo_modeldir,conf=0.4,iou=0.4):
    model=torch.hub.load(yolo_modeldir,'yolov5l6', source='local',pretrained=True)

    model.conf = conf
    model.iou = iou
    model.max_det = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda:",torch.cuda.is_available())
    video_model = slowfast_r50_detection(True).eval().to(device)
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/ava_action_list.pbtxt")
    return model,video_model,deepsort_tracker,ava_labelnames
def yolo_dectection(model,framelist):
    global yolo_preds
    model.conf=0.4
    yolo_preds = model(framelist,size=640)
    show_yolopreds(yolo_preds)

def yolo_slowfast_action(model,deepsort_tracker,framelist):

    model.conf=0.4
    imsize=640
    yolo_preds = model(framelist,size=imsize)
    show_yolopreds(yolo_preds)
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
    save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map,saveimg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo5l6=r'C:\Users\Administrator\.cache\torch\hub\ultralytics_yolov5_master'
webcam_ip='rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream'
# input = "demo/B6-8-23-10-46-t.mp4"
yolo_model,slowfast_model,deepsort_tracker,ava_labelnames=load_model(yolo5l6)
if not os.path.exists('./demo/result'):
    os.mkdir('./demo/result')
ctime=time.ctime()
today_time=ctime.replace(' ','_').replace(':','_')
todaya_time_folder='./demo/result/'+today_time
if not os.path.exists(todaya_time_folder):
    os.mkdir(todaya_time_folder)
frame_list=[]
cap=cv2.VideoCapture(0)
global yolo_preds
success, frames = cap.read()
def receive():
    global frame_list
    success,frames=cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('Cam ok fps:{}'.format(fps))
    # q.put(frames)
    frames_num=0
    while success:
        success, frames = cap.read()
        # imgs.append(frames)
        if frames_num%1==0:
            q.append(frames)
        if len(q) >= (maxsize):
            frame_list=list(q)

        if frames_num>=24:
            frames_num=0
        frames_num+=1
    cv2.VideoCapture(0).release()
def yolo_process():
    time.sleep(2)
    while True:

        yolo_slowfast_action(yolo_model, deepsort_tracker, frame_list)
# def action_process():
#     global yolo_preds
#     time.sleep(3)
#     while True:
#         action_recognization(yolo_preds, deepsort_tracker, frame_list)

def show_result():

    while True:
        if (p_result.full() != True):
            # print("show_result.................")
            result_frame = p_result.get()
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            print("display_invalid:",p_result.empty())


app = Flask(__name__)
cond=threading.Condition()

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(show_result(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
if __name__ == "__main__":
    q1=threading.Thread(name='receive',target=receive)
    q2=threading.Thread(name='yolo_dectection',target=yolo_process)
    # q3=threading.Thread(name='action_recognization',target=action_process)
    q4=threading.Thread(name='show_result',target=app.run)
    q1.start()
    q2.start()

    q4.start()

    # app.run(debug=False)
