import cv2
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 10


def visualize(helmet_df, game_play, view, frame, down_ratio=2):

    image_dir = f'../input/train_frames_half/{game_play}_{view}/'
    img_path = os.path.join(image_dir, f'{frame:06}.jpg')
    img = cv2.imread(img_path)[:, :, ::-1].copy()

    frame_helmets = helmet_df.query('game_play == @game_play & frame == @frame & view==@view')
    for bbox in frame_helmets[['left', 'top', 'width', 'height']].values:
        bbox //= down_ratio
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)
    plt.imshow(img)
