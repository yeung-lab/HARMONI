import os

import cv2
import requests
from pytube import YouTube
from PIL import Image
import imageio

image_exts = ['.jpg', '.jpeg', '.png']

def download_and_process_seedlings_sample(save_path, fps=1):
    if not os.path.exists(save_path):
        url = 'https://bergelsonlab.com/seedlings/images/30sMB.mp4'  # this is public at https://bergelsonlab.com/seedlings/
        response = requests.get(url)
        response.raise_for_status()  # Check for any errors during the request
        with open(save_path, 'wb') as file:
            file.write(response.content)

    vid_name = os.path.splitext(os.path.basename(save_path))[0]
    out_dir = os.path.join(os.path.split(save_path)[0], vid_name+'_fps'+str(fps))
    os.makedirs(out_dir, exist_ok=True)

    coord = [100,440,360,960]  # main view coord: y1,y2,x1,x2
    vidcap = cv2.VideoCapture(save_path)
    video_fps = round(vidcap.get(cv2.CAP_PROP_FPS))  # 30 for seedlings
    print('Original video has fps', video_fps, '. Downsampling to fps', fps)
    count = 0
    while True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, count*int(1000/fps))
        success, image = vidcap.read()
        if not success: break
        # only keep the main view (i.g. take out the head cam views)
        cropped_image = image[coord[0]:coord[1],coord[2]:coord[3]]
        cv2.imwrite(os.path.join(out_dir, "frame_%08d.jpg" % count), cropped_image)
        count += 1


def download_youtube_video(save_path, video_id='aWV7UUMddCU'):
    # TODO: not working. something wrong with pytube.
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    print(video_url)
    yt = YouTube(video_url)
    yt.streams.get_highest_resolution().download(output_path=save_path, filename="youtube.mp4")
    

def mp4_to_images(file_path, save_path, start_frame=0, end_frame=100, fps=1):
    os.makedirs(save_path, exist_ok=True)

    vidcap = cv2.VideoCapture(file_path)
    video_fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    print('Original video has fps', video_fps, '. Downsampling to fps', fps)
    every_x_frame = video_fps // fps
    frame_cnt = 0; img_cnt = 0
    while vidcap.isOpened():
        success,image = vidcap.read()
        if not success: break
        if frame_cnt % every_x_frame == 0 and frame_cnt > start_frame and frame_cnt < end_frame:
            cv2.imwrite(os.path.join(save_path, "frame_%08d.jpg" % img_cnt), image)
            img_cnt += 1
        frame_cnt += 1
    vidcap.release()
    cv2.destroyAllWindows()


def gif_to_images(gif_path, output_path):
    with Image.open(gif_path) as im:
        frames = []
        for frame_index in range(im.n_frames):
            im.seek(frame_index)
            frames.append(im.copy().convert("RGB"))

    for index, frame in enumerate(frames):
        frame.save(os.path.join(output_path, "frame_%08d.png" % index), **frame.info)


def images_to_gif(image_dir, gif_path, fps):
    images = []
    for filename in sorted(os.listdir(image_dir)):
        if os.path.splitext(filename)[1] in image_exts and not filename.startswith("."):
            images.append(imageio.imread(os.path.join(image_dir, filename)))
    
    # `fps=50` == `duration=20` (1000 * 1/50).
    duration = 1000 * 1/fps
    imageio.mimsave(gif_path, images, duration=duration)


def repeat_gif(gif_path, output_path, num_repeats):
    with imageio.get_reader(gif_path) as reader:
        frames = [frame for frame in reader]

    repeated_frames = frames * num_repeats

    with imageio.get_writer(output_path, mode='I', loop=0) as writer:
        for frame in repeated_frames:
            writer.append_data(frame)


def images_to_mp4(image_dir, vid_path, fps):
    image_height, image_width, _ = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[0])).shape
    vid = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_width, image_height))
    for filename in sorted(os.listdir(image_dir)):
        if os.path.splitext(filename)[1] in image_exts and not filename.startswith("."):
            image = cv2.imread(os.path.join(image_dir, filename))
            vid.write(image)
    vid.release()


def video_to_images(vid_path, save_path, **kwargs):
    ext = os.path.splitext(vid_path)[1]
    if ext == '.gif':
        gif_to_images(vid_path, save_path)
    elif ext == '.mp4':
        mp4_to_images(vid_path, save_path, **kwargs)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    # download_and_process_seedlings_sample('./data/demo/seedlings.mp4', fps=1)
    repeat_gif('./teasers/video.gif', './teasers/video_repeated.gif', num_repeats=3)