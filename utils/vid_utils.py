from pytube import YouTube

def download_video(save_path, video_id='aWV7UUMddCU'):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    print(video_url)
    yt = YouTube(video_url)
    # video = yt.streams.get_highest_resolution()  # Get the highest resolution stream
    # video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    # video.download(output_path=save_path)
    yt.streams.get_highest_resolution().download(output_path=save_path, filename="youtube.mp4")
    
if __name__ == '__main__':
    download_video('./data/demo')
