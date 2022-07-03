import logging 
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from tqdm import tqdm

from utils.constant import HOMEPATH, DP_JUMP, TASK_BLACK_FRAME, TASK_NO_PERSON
from data.visualize import visualize_one_frame
from tools.detection import load_yolo5_model

logger = logging.getLogger(__name__)
LOG_TEMPLATE = "\n"+"-"*10+" %s "+"-"*10

def load_frames_from_path(
  video_path:str, # full path to video to load 
  enabled_tasks:list=[TASK_BLACK_FRAME,TASK_NO_PERSON], # list of frame types to filter out. currently support black frame & no person
  save_to_folder=False, # pickle save to local 
  folder_path:str = None, # full path to local folder to save
  ):
  """
  Args:
    video_path: full path to video to load 
    enabled_task: list of filtering tasks. Currently support black frame and no person frame 
    folder_path: full path to local folder if save_to_folder is True

  Returns:
    frames: Dict with entries 
      "good": frames that passed all the filtering tasks 
      TASK_x: frames that was filtered by 
  """

  # load video.
  logger.warning(LOG_TEMPLATE,f"loading video from path {video_path}.")
  if not os.path.exists(video_path):
    raise FileNotFoundError(f"{video_path} does not exist.")
  
  # get frames 
  count = 0
  all_frames=[]

  cap=cv2.VideoCapture(video_path)
  while(cap.isOpened()):
      count += 1
      ret, frame = cap.read()
      if ret == False: break
      if count % DP_JUMP!= 0: continue 
      all_frames.append(frame)
  cap.release()
  cv2.destroyAllWindows()

  # filter by task 
  frames = {}
  frames['good']=[]
  for task_name in enabled_tasks:
    frames[task_name]=[]
  
  if TASK_NO_PERSON in enabled_tasks:
    yolo = load_yolo5_model()

  for frame in all_frames:
    # remove black frames 
    if TASK_BLACK_FRAME in enabled_tasks:
      if np.average(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)) < 12:
        frames[TASK_BLACK_FRAME].append(frame)
        continue 

    # remove frame without people 
    if TASK_NO_PERSON in enabled_tasks:
      predictions = yolo(frame).pred[0]
      if not 0 in predictions[:, 5]:  # person label is 0
        frames[TASK_NO_PERSON].append(frame)
        continue 
    
    # good frames 
    frames['good'].append(frame)
  
    
  # save to local 
  if save_to_folder:
    if not os.path.exists(folder_path):
      raise FileNotFoundError(f"{folder_path} does not exist.")
    logger.warning(LOG_TEMPLATE,f"Saving frames to {folder_path}.")
    raise NotImplementedError()

  return frames



def pull_person_bbox(frames):
  """
  (for centroid method) pull bounding box regions for each person 
  in each frame. 

  Args:
    frames: np array, all frames need to contain at least one person TODO can relax 
  
  Returns:
    gallary: masked np array, same size as input. 

  """

  logger.warning(LOG_TEMPLATE,"pulling gallery bbox.")
  model = load_yolo5_model()
  gallery = []
  parent_mask = np.zeros(frames[0].shape,dtype=np.uint8)

  for frame in tqdm(frames):
    pred = model(frame).pred[0]
    boxes = pred[:, :4] # x1, y1, x2, y2
    categories = pred[:, 5]
    if not 0 in categories:
        logger.warning(LOG_TEMPLATE,"WARNING: contains empty frame.")
        continue 
    
    idxs = np.where(categories==0)
    for idx in idxs:
      box = boxes[idx]
      mask = cv2.rectangle()
