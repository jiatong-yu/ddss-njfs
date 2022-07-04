from data.dataprep import load_frames_from_path,pull_person_bbox
from data.visualize import visualize_one_frame
from utils.constant import HOMEPATH


def _test_pull_person(vid_path):
    all_frames = load_frames_from_path(vid_path)
    frames = all_frames['good']
    pulled_bbox = pull_person_bbox(frames)
    for bbox in pulled_bbox:
        visualize_one_frame(bbox)
    return 


default_path = HOMEPATH+"data/test_vid.mp4"
_test_pull_person(default_path)