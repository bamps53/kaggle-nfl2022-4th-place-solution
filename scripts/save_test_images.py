import argparse
import os
import glob
import cv2
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

START_STEP = 290
SAVE_DIR = '/temp/test_frames'


def save_images(video_path):
    video_name = os.path.basename(video_path).split('.')[0]
    save_dir = f'{SAVE_DIR}/{video_name}'
    os.makedirs(save_dir, exist_ok=True)

    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    num_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Write Frames
    cnt = 0
    with tqdm(range(int(num_frames))) as pbar:
        while video_cap.isOpened():
            success, image_frame = video_cap.read()
            if not success:
                break
            save_path = f"{save_dir}/{cnt:06}.jpg"
            if cnt >= START_STEP and not os.path.exists(save_path):
                image_frame = cv2.resize(image_frame, (int(width), int(height)))
                cv2.imwrite(save_path, image_frame)
            cnt += 1
            pbar.update()


if __name__ == "__main__":
    video_paths = sorted(glob.glob('../input/nfl-player-contact-detection/test/*ne.mp4'))
    print('num videos:', len(video_paths))

    pool = Pool(processes=cpu_count())
    with tqdm(total=len(video_paths)) as t:
        for _ in pool.imap_unordered(save_images, video_paths):
            t.update(1)
