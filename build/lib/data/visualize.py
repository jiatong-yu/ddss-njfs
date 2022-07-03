
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


def visualize_one_frame(frame, boxes=None, labels=None):
    """
    visualize one frame sample and dump into "./temp_vis.png"
    Args:
        boxes: bounding boxes in list. If passing only one 
                bbox, pass it as [box]
        labels: corresponding to bbox, have to be same length. 
                
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.imshow(frame[...,::-1])

    if boxes is not None:
        for box in boxes:
            rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='turquoise')
            ax2.add_patch(rect)
            
    plt.savefig('temp_vis.png')
    return 