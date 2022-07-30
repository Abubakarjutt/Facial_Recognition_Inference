
#!pip install face-recognition==0.2.0 (this version for cpu otherwise use latest)
#!pip install face-detection

import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger
import json
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
#from yolox.utils.visualize import plot_tracking
#from yolox.tracker.byte_tracker import BYTETracker
from byte_tracker import *
from yolox.tracking_utils.timer import Timer
import cv2
import numpy as np
import glob
import os
#from deepface import DeepFace
import matplotlib.pyplot as plt
#from deepface import DeepFace
from numpy import asarray
from PIL import Image
# from mtcnn import MTCNN
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
# from tensorflow.keras.layers import Concatenate
# from tensorflow.keras.layers import Lambda, Flatten, Dense
# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.layers import Layer
# from tensorflow.keras import backend as K
# K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
# import tensorflow as tf
import PIL
import face_recognition
import face_detection
import pickle
from datetime import datetime
import distutils.dir_util


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False

    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

class Face_rec:
  def __init__(self, video='data/palace.mp4',
  frameps=1,
  demo='video',
  fps=30,
  exp="exps/example/mot/yolox_x_mix_det.py",
  weight='pretrained/bytetrack_x_mot17.pth.tar',
  save=True,
  v_path=None,
  fp16=False,
  thurshold=0.51,
  device='cpu',
  registered_faces = 'database.pkl'):
    self.path=video
    self.exp=exp
    self.MAX_DISTANCE=thurshold
    self.v_path=v_path
    self.demo =demo
    self.frameps=frameps
    self.camid=0
    self.ckpt=weight
    self.save_result=True
    self.IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
    self.detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.4, nms_iou_threshold=.3)
    self.database = open(registered_faces, "rb")
    self.database = pickle.load(self.database)
    self.experiment_name=None
    self.expn=None
    self.name=None
    self.save_result=True
    self.exp_file=exp
    self.device=device
    self.conf=None
    self.nms=None
    self.tsize=None
    self.fps=fps
    self.fp16=fp16
    self.fuse=False
    self.trt=False
    self.track_thresh=0.5
    self.track_buffer=30
    self.match_thresh=0.8
    self.aspect_ratio_thresh=1.6
    self.min_box_area=10
    self.mot20=False
    self.trt_file=None
    self.decoder=None

    self.exp = get_exp(self.exp_file, self.name)
    if not self.experiment_name:
        self.experiment_name = self.exp.exp_name

    self.output_dir = osp.join(self.exp.output_dir, self.experiment_name)
    os.makedirs(self.output_dir, exist_ok=True)

    if self.save_result:
        self.vis_folder = osp.join(self.output_dir, "track_vis")
        os.makedirs(self.vis_folder, exist_ok=True)

    # if self.trt:
    #     self.device = "gpu"
    self.device = torch.device('cpu')

    #logger.info("Args: {}".format(args))

    if self.conf is not None:
        self.exp.test_conf = self.conf
    if self.nms is not None:
        self.exp.nmsthre = self.nms
    if self.tsize is not None:
        self.exp.test_size = (self.tsize, self.tsize)

    self.model = self.exp.get_model().to(self.device)
    logger.info("Model Summary: {}".format(get_model_info(self.model, self.exp.test_size)))
    self.model.eval()

    if not self.trt:
        if self.ckpt is None:
            ckpt_file = osp.join(self.output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = self.ckpt
        logger.info("loading checkpoint")
        self.ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        self.model.load_state_dict(self.ckpt["model"])
        logger.info("loaded checkpoint done.")

    if self.fuse:
        logger.info("\tFusing model...")
        self.model = fuse_model(self.model)

    if self.fp16:
        self.model = self.model.half()  # to FP16

    if self.trt:
        assert not self.fuse, "TensorRT model is not support model fusing!"
        self.trt_file = osp.join(self.output_dir, "model_trt.pth")
        assert osp.exists(
            self.trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        self.model.head.decode_in_inference = False
        decoder = self.model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    self.predictor = Predictor(self.model, self.exp, self.trt_file, self.decoder, self.device, self.fp16)
    self.current_time = time.localtime()

    #if self.demo == "image":
    #    image_demo(self.predictor, vis_folder, current_time, args)
    #elif self.demo == "video" or args.demo == "webcam":
    #    imageflow_demo(self.predictor, vis_folder, current_time, args)
  def get_location(self,detections):
      thresh=0.4
      inds = np.where(detections[:, -1] >= thresh)[0]
      b=[]
      for h in range(len(inds)):
          inner = []
          bbox = detections[h, :4]
          inner.append(int(bbox[1]))
          inner.append(int(bbox[2]))
          inner.append(int(bbox[3]))
          inner.append(int(bbox[0]))
          b.append(inner)
      return b
  def get_face_embeddings_from_image(self,image,face_locations, convert_to_rgb=False):
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings
  def get_image_list(self):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(self.path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in self.IMAGE_EXT:
                image_names.append(apath)
    return image_names
  def write_results(self,filename, results):
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids, scores in results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                    f.write(line)
        logger.info('save results to {}'.format(filename))

  def imageflow_demo(self):
        cap = cv2.VideoCapture(self.path if self.demo == "video" else self.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", self.current_time)
        save_folder = osp.join(self.vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        if self.demo == "video":
            save_path = osp.join(save_folder, self.path.split("/")[-1])
        else:
            save_path = osp.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")

        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        tracker = BYTETracker(self, frame_rate=fps)
        timer = Timer()
        frame_id = 0
        results = []
        res2=[]
        #MAX_DISTANCE = 0.51
        known_face_encodings = list(self.database.values())
        known_face_names = list(self.database.keys())
        #print(known_face_names)
        #addr=self.path
        #addr=addr.replace(".mp4", "")
        #if(self.v_path=='None'):
        try:
          os.makedirs('/content/output/')
        except:
          pass
        now=datetime.now().strftime('%Y-%m-%d')
        if(self.v_path is None):
                                    #os.makedirs('/content/output/'+str(current_time)+"/", exist_ok=True)
                                    v_path2='/content/output/'+str(now)+"/"
                                    distutils.dir_util.mkpath(v_path2)
        else:
                                    #os.makedirs(self.v_path+"/", exist_ok=True)
                                    v_path2=self.v_path+str(now)+"/"
                                    distutils.dir_util.mkpath(v_path2)

        check=False
        while True:
            if frame_id % 10 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            ret_val, frame = cap.read()
            #print(frame.shape)
            if (frame_id < length-1):
             try:
              if(frame_id%self.frameps==0):
                outputs, img_info = self.predictor.inference(frame, timer)
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                            x1, y1, w, h = tlwh
                            intbox = tuple(map(int, (y1, y1+h ,x1, x1 + w)))
                            #print(intbox)
                            test=frame[intbox[0]:intbox[1], intbox[2]:intbox[3]]
                            #cv2.imwrite("person.jpg",test)
                            #test = extract_face_from_image(test)
                            #df = DeepFace.find(test, db_path = "/content/db",model_name = 'Facenet512',enforce_detection=True,detector_backend= 'dlib',)


                            ##result = find_match(test)
                            test = test[:, :, ::-1]
                            detections = self.detector.detect(test)
                            location = Face_rec.get_location(self,detections)
                            if(len(location)==0):
                              result=False
                            else:
                              #top,right, bottom,left = location[0]
                              #test=test[top:bottom,left:right]
                              try:
                                #result = who_is_it(test, database, FRmodel)
                                # run detection and embedding models
                                face_locations, face_encodings = Face_rec.get_face_embeddings_from_image(self,test,location, convert_to_rgb=True)

                                # Loop through each face in this frame of video and see if there's a match
                                for location, face_encoding in zip(face_locations, face_encodings):

                                    # get the distances from this encoding to those of all reference images
                                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                    #print("here")


                                    # select the closest match (smallest distance) if it's below the threshold value
                                    if np.any(distances <= self.MAX_DISTANCE):
                                        result=True
                                        #print("found")
                                    else:
                                        result = False
                              except:
                                result=False
                                #print(df)
                            #print(result)
                            #sdfsf

                              # select the closest match (smallest distance) if it's below the threshold value
                            try:
                              if (result):
                                  tid = "Known"
                                  online_tlwhs.append(tlwh)
                                  online_ids.append(tid)
                                  online_scores.append(t.score)
                                  results.append(
                                      f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                  )
                              else:
                                  tid = "Unknown"
                                  online_tlwhs.append(tlwh)
                                  online_ids.append(tid)
                                  online_scores.append(t.score)
                                  results.append(
                                      f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                  )
                                  #now = datetime.now()
                                  #current_time = now.strftime("%H:%M:%S")
                                  time_dict = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                                  try:
                                      fpath=v_path2+str(frame_id)+".jpg"
                                  except:
                                      fpath=v_path2+'/'+str(frame_id)+".jpg"
                                  check=True
                                  #cv2.imwrite(fpath, online_im)
                                  temp={'Frame':frame_id,'class':tid,'time':time_dict,'frame_path':fpath}
                                  res2.append(temp)

                            except:
                                  online_tlwhs.append(tlwh)
                                  online_ids.append(tid)
                                  online_scores.append(t.score)
                                  results.append(
                                      f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                  )
                                  continue
                    timer.toc()
                    online_im = Face_rec.plot_tracking(self,
                        img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )
                    if check:
                      #print("stored")
                      cv2.imwrite(fpath, online_im)
                      check=False
                    #cv2.imwrite("0.jpg",online_im)
                else:
                    timer.toc()
                    online_im = img_info['raw_img']
                #if self.save_result:
                #    vid_writer.write(online_im)
                #print("uncomment these two lines to save the video")
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
             except:
                 frame_id += 1

                 continue
            else:
                break
            #count=2
            #cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            frame_id += 1



        if self.save_result:
            res_file = osp.join(self.vis_folder, f"{timestamp}.txt")
            res_json = osp.join(self.vis_folder, f"{timestamp}.json")
            with open(res_file, 'w') as f:
                f.writelines(results)
            a_file = open(res_json, "w")
            json.dump(res2, a_file)
            a_file.close()
            logger.info(f"save results to {res_file}")
        return res_file,save_path,res2

  def plot_tracking(self,image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = (obj_ids[i])
        id_text = '{}'.format((obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        if(obj_id=='Known'):
          color = (0, 128, 0)
          cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
          cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 128, 0) ,
                    thickness=text_thickness)
        else:
          color = (0, 0, 255)
          cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
          cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

from inference2 import *
detector = Face_rec(video='k1_unk1_3.mp4',v_path='/',device='cpu')
a, b, c = detector.imageflow_demo()
print(c)
