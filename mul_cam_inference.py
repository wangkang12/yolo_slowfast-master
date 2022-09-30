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
from xlutils.copy import copy
import os
import time
import xlwt,xlrd

import cv2
import numpy as np
import multiprocessing as mp
maxsize=2
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
                 velocity=None, thickness=2, fontsize=0.7, fontthickness=2):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, int(thickness), lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, 2)
    cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [128, 0, 128], fontthickness)
    return img

def IOU(box1,box2):
    (x1,y1),(x2,y2)=box1
    (x3,y3),(x4,y4)=box2
    x_inter1=max(x1,x3)
    y_inter1=max(y1,y3)
    x_inter2=min(x2,x4)
    y_inter2=min(y2,y4)
    if (x_inter2>x_inter1) and (y_inter2>y_inter1):
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2 - y1)
        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
    else:
        iou=0
    return iou


def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs



def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map,saveimg=False):
    img_num = len(yolo_preds.ims)
    img_label=yolo_preds.names
    save_img_actionlabel = ["bend/bow","crawl", "crouch/kneel", "getup/squat", "carry/hold", "climb","smoke",'fight/hit','touch']
    save_img_label=['backpack','handbag','suitcase']
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        if ((i>=int((img_num-8))) and (pred.shape[0])):
            # yolo_imglabel=[]
            # slow_fast_label=[]
            person_box=[]
            packages_box=[]
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    yolo_label = img_label[int(cls)]
                    ava_label=' '
                    if yolo_label in save_img_label:
                        # yolo_imglabel.append(yolo_label)
                        packages_box.append(box)
                    # continue
                elif trackid in id_to_ava_labels.keys():
                    yolo_label = 'person'
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                    if ava_label in save_img_actionlabel:
                        # slow_fast_label.append(ava_label)
                        person_box.append(box)
                else:
                    yolo_label = 'Unknow'
                    ava_label = 'Unknow'
                # if (int(cls) != 0):#just only detecting the person
                #     continue
                print("avalabel:***{}***".format(ava_label))
                text = '{} {}'.format(yolo_label,ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box, im, color, text)
            if saveimg and (len(person_box)>0)and(len(packages_box)>0):
                for person_b in person_box:
                    for package_b in packages_box:
                        box1 = (tuple(person_b[:2]), tuple(person_b[2:]))
                        box2 = (tuple(package_b[:2]), tuple(package_b[2:]))
                        iou = IOU(box1, box2)
                        if iou>0.001:
                            imgname = time.ctime()
                            randint = np.random.randint(100)
                            imgname = imgname.replace(' ', '_').replace(':', '_')
                            cv2.imwrite(todaya_time_folder + '//' + str(imgname) + '_' + str(randint) + '.jpg', im)
            AI_queue.put(im)
            time.sleep(0.01)
            AI_queue.get() if AI_queue.qsize() > 1 else time.sleep(0.000001)




def show_yolopreds(yolo_preds,AI_queue,cam_name_queue,project_dir,saveimg=False):
    # img_num = len(yolo_preds.ims)
    save_img_label=['backpack','handbag','suitcase']
    save_img_label = ['backpack', 'handbag', 'suitcase','chair','cup']#测试
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        # if i >= int((img_num - 0)):
        #     continue
        imageIndex = yolo_preds.pandas().xyxy[i]
        person_label=[]
        img_label=[]
        if pred.shape[0]:
             # img1 predictions (pandas)
            if imageIndex.shape[0]:
                for box_num in range(int(imageIndex.shape[0])):
                    startX = int(imageIndex["xmin"][box_num])
                    startY = int(imageIndex["ymin"][box_num])
                    endX = int(imageIndex["xmax"][box_num])
                    endY = int(imageIndex["ymax"][box_num])
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    if imageIndex["name"][box_num]=='person':
                        person_label.append([(startX, startY), (endX, endY)])
                    if imageIndex["name"][box_num] in save_img_label:
                        img_label.append([(startX, startY), (endX, endY)])
                    im=cv2.putText(im, imageIndex["name"][box_num],(startX, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
                    im=cv2.rectangle(im,(startX, startY), (endX, endY), (0, 255, 0), 2)
        if saveimg:
            # label_person=imageIndex["name"].tolist()
            # print(label_person)
            if 'person' in imageIndex["name"].tolist():
                for label in imageIndex["name"].tolist():
                    if label=='person':
                        continue
                    elif label in save_img_label:
                        for package_box in img_label:#IOU
                            for person_box in person_label:
                                # print('person_label,img_label',person_label,img_label)
                                iou=IOU(package_box[:],person_box[:])
                                print('iou____________________________:',iou)
                                if iou>0.0:
                                    print("save imgs:bad behave............")
                                    imgname = time.ctime()
                                    imgname = imgname.replace(' ', '_').replace(':', '_')
                                    img_int=np.random.randint(1000)
                                    cv2.imwrite(project_dir + '//' + str(imgname) + str('_yolo_') + str(img_int) + '.jpg', im)
                                    # cv2.imwrite(todaya_time_folder + '//' + str(imgname) + '.jpg', im)
                                    if not cam_name_queue.empty():
                                        cam_name=cam_name_queue.get()
                                        cam_name_queue.put(cam_name)
                                        # print(cam_name,"99999999999999999999999999999999999999999999999999999")
                                        write_log(project_dir, cam_name, imgname, label,place='未知')
                                    #写入excel
                    else:
                        continue
            else:
                print('NO Person!')

        AI_queue.put(im)
        time.sleep(0.1)########非常重要设置参数等待取
        if AI_queue.qsize() >=2:
            ims=AI_queue.get()
        else:
            AI_queue.put(ims)

def load_model(yolo_modeldir,conf=0.4,iou=0.4):

    yolo_model=torch.hub.load(yolo_modeldir,'yolov5l6', source='local',pretrained=True)

    yolo_model.conf = conf
    yolo_model.iou = iou
    yolo_model.max_det = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    print("cuda:",torch.cuda.is_available())
    # # video_model = slowfast_r50_detection(True).eval().to(device)
    # slowfast_model = slowfast_r50_detection(True).eval().to(device)
    # deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    # ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/ava_action_list.pbtxt")

    return yolo_model


def yolo_slowfast_action(model,framelist,AI_queue,cam_name_queue,project_dir):
    # print(len(framelist),framelist)
    model.conf=0.65
    imsize=640
    yolo_preds = model(framelist,size=imsize)
    show_yolopreds(yolo_preds,AI_queue,cam_name_queue,project_dir,saveimg)


def image_put(input_queue,cam_ip,cam_name_queue):
    # camera_ip='./demo/B6-8-23-10-46-t.mp4'
    camera_ip=cam_ip
    if len(camera_ip)==1:
        camera_ip=int(camera_ip)
    cap = cv2.VideoCapture(camera_ip)
    is_opened, frame = cap.read()
    print('cam open:',is_opened)
    while is_opened:
        is_opened, frame = cap.read()
        # if is_opened==False:
        #     frame= np.ones((512, 512, 3), np.uint8)
        input_queue.put(frame)
        cam_name_queue.put(camera_ip)
        cam_name_queue.get() if input_queue.qsize() >=2 else None
        # print(input_queue.qsize())
        # time.sleep(0.05)
        # input_queue.get() if input_queue.qsize() >=3 else None
        input_queue.get() if input_queue.qsize() >= 3 else None
        time.sleep(0.01)
    print('cam closed!')
    cv2.VideoCapture(camera_ip).release()

def show_result(flaskresult_queue):
    while True:
        # time.sleep(0.3)
        if not flaskresult_queue.empty():
            print("show_result.................")
            try:
                result_frame = flaskresult_queue.get()
            except:
                time.sleep(1)
                result_frame = flaskresult_queue.get()
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            continue
            print("display_invalid:",flaskresult_queue.empty())

def AI_process(queue,AI_queue,cam_name_queue,project_dir):
    yolo_model= load_model(yolo5l6)
    while True:
        print('len_queue:',queue.qsize())
        if not queue.empty():
            # print('len_queue')
            frame_list=[queue.get()]
            # print(frame_list)
            try:
                yolo_slowfast_action(yolo_model,frame_list,AI_queue,cam_name_queue,project_dir)
            except:
                continue
        else:
            time.sleep(0.1)
            print("Wait Img")
def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

def image_collect(queue_list,flaskresult_queue):
    time.sleep(1)
    show_imgsie=512
    img1= np.ones((show_imgsie,show_imgsie,3), np.uint8)
    # print('img1',img1.shape)

    temp_img=[img1,img1,img1]
    while True:
        for cam_num in range(len(queue_list)):
            if not queue_list[cam_num].empty():
                # time.sleep(1)
                img=queue_list[cam_num].get()
                # print(img.shape)
                temp_img[cam_num]=img
                queue_list[cam_num].put(img)
            else:
                queue_list[cam_num].put(temp_img[cam_num])
                # queue_list[cam_num].put(temp_img[cam_num])
        imgs = [cv2.resize(q.get(), (show_imgsie, show_imgsie)) for q in queue_list]
        imgs = np.concatenate(imgs, axis=1)
        flaskresult_queue.put(imgs)
        flaskresult_queue.get() if flaskresult_queue.qsize() >= 2 else time.sleep(0.05)



app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(show_result(flaskresult_queue), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
def set_style(name,height,bold=False):
    style = xlwt.XFStyle()# 初始化样式，
    font = xlwt.Font() # 为样式创建字体
    font.name = name#字体样式
    font.bold = bold# 粗体
    font.underline = True  # 下划线
    font.italic = True  # 斜体字
    font.colour_index = 2
    font.height = height #字高
    style.font = font
    return style

#写Excel
def make_excel(excel_dir):
    f = xlwt.Workbook(encoding='ascii')#创建对象，指定编码为'ascii'
    sheet1 = f.add_sheet('检测记录',cell_overwrite_ok=True)
    row0= ["摄像头","时间","行为","地点"]
    for i in range(0,len(row0)):
        sheet1.write(0,i,row0[i],set_style('Times New Roman',220,False))# sheet1.write（行，列，值，格式）
    style = xlwt.XFStyle()
    style.num_format_str = 'D-MMM-YY'  # 设置单元格时间格式Other options: D-MMM-YY, D-MMM, MMM-YY, h:mm, h:mm:ss, h:mm, h:mm:ss, M/D/YY h:mm, mm:ss, [h]:mm:ss, mm:ss.0
    f.save(excel_dir+'//test_log.xls')
def write_log(project_dir,cam_name,times,behave,place='未知'):
    old_workbook=xlrd.open_workbook(project_dir+'//test_log.xls','a')
    new_workbook= copy(old_workbook)  # 复制
    log_list=[cam_name,times,behave,place]
    sheetName= old_workbook.sheet_names()  # 取sheet表
    sheets=old_workbook.sheet_by_name(sheetName[0])
    rows=sheets.nrows
    print(rows)
    cols=len(log_list)
    new_sheet=new_workbook.get_sheet(0)
    for col in range(cols):
        new_sheet.write(rows, col, log_list[col])
    new_workbook.save(project_dir+'\\'+'test_log.xls')
def mkdir_file():
    global project_dir
    if not os.path.exists('./demo/result'):
        os.mkdir('./demo/result')
    ctime = time.ctime()
    today_time = ctime.replace(' ', '_').replace(':', '_')
    todaya_time_folder = './demo/result/' + today_time
    project_dir=todaya_time_folder
    if not os.path.exists(todaya_time_folder):
        os.mkdir(todaya_time_folder)
        make_excel(todaya_time_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo5l6 = r'C:\Users\Administrator\.cache\torch\hub\ultralytics_yolov5_master'    #linux修改地址
# yolo5l6=r'/home/foxconnai/.cache/torch/hub/ultralytics_yolov5_master'

# mp.set_start_method('forkserver',force = True) #Linux系统设置

def run_multi_camera(project_dir,rtsp='rtmpCollection.txt'):
    rtsp_list=[]
    with open(rtsp,'r') as rtspfile:
        rtsps=rtspfile.readlines()
        for rtsp_num in range(len(rtsps)):
            rtsp_ip=rtsps[rtsp_num].strip()
            rtsp_list.append(rtsp_ip)

    print(rtsp_list)
    # for cam_ip in rtsp_list:
    #     cam_ip_queue.put(cam_ip)
    frame_queues = [mp.Queue(maxsize=4) for _ in rtsp_list]
    AI_result_queue=[mp.Queue(maxsize=4) for _ in rtsp_list]
    camip_queue=[mp.Queue(maxsize=4) for _ in rtsp_list]
    # processes = [mp.Process(target=image_collect, args=(AI_result_queue,flaskresult_queue))]
    processes=[]
    for in_queue,AI_result,cam_ip,cam_name_queue in zip(frame_queues,AI_result_queue,rtsp_list,camip_queue):
        # global input_queue
        # input_queue=in_queue
        processes.append(mp.Process(target=image_put, args=(in_queue,cam_ip,cam_name_queue)))
        processes.append(mp.Process(target=AI_process, args=(in_queue,AI_result,cam_name_queue,project_dir)))
    processes.append(mp.Process(target=image_collect,args=(AI_result_queue,flaskresult_queue)))
    # processes.append(mp.Process(target=app.run(port=5555)))
    for process in processes:
        process.daemon = True
        process.start()
    # processes.append(mp.Process(target=app.run('10.107.122.114',5000)))  #linux 修改
    processes.append(mp.Process(target=app.run()))

    processes[-1].daemon = True
    processes[-1].start()
    for process in processes:
        process.join()

if __name__=='__main__':
    mp.set_start_method('spawn', force=True)
    project_dir = ''
    cam_folder_file=threading.Thread(name='make_folder_file', target=mkdir_file).start()
    # AI_queue = mp.Queue(maxsize=4)
    # input_queue = mp.Queue(maxsize=4)
    # print(project_dir)
    flaskresult_queue = mp.Queue(maxsize=4)
    run_multi_camera(project_dir,rtsp='rtmpCollection.txt')
