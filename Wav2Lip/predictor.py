import os
import subprocess
import cv2
import face_alignment
import torch
from pydub import AudioSegment

from . import predictor_api as api
from .models import Wav2Lip


# Initialize face detection and Wav2Lip models
def init(wav2lip_checkpoint_path, device="cpu"):
    print("Initializing face and landmark detection model")
    face_alignment_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, verbose=True,
                                            face_detector='blazeface')

    print("Initializing Wav2Lip checkpoint from: {}".format(wav2lip_checkpoint_path))
    wav2lip_model = Wav2Lip().to(device)
    if device == 'cuda':
        checkpoint = torch.load(wav2lip_checkpoint_path)
    else:
        checkpoint = torch.load(wav2lip_checkpoint_path, map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    wav2lip_model.load_state_dict(new_s)

    wav2lip_model = wav2lip_model.eval()
    return face_alignment_detector, wav2lip_model


# function to generate video frames to align with the audio
def predict(face_alignment_detector, wav2lip_model, images_cv, audio_pydub, video_fps,
            device="cpu",
            mouth_mask_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/mouth_mask.npy'),
            face_det_batch_size=10,
            face_det_confidence_score_min=0.80,
            wav2lip_batch_size=10,
            pads=[0,0,0,0]):

    data = api.preprocess_data(face_alignment_detector, images_cv, audio_pydub, video_fps, face_det_batch_size, face_det_confidence_score_min, pads)
    predictions = api.process_in_batches(data, wav2lip_batch_size,
                                         lambda x: api.process_wav2lip_batch(x, wav2lip_model, device=device))
    processed_frames = api.postprocess_wav2lip(images_cv, predictions, data, mouth_mask_path)

    return processed_frames

def repl_test():
    from codetiming import Timer
    wav2lip_checkpoint_path = './wav2lip_gan.pth'
    device = "cpu"
    mouth_mask_path = './data/mouth_mask.npy'

    # init
    face_alignment_detector, wav2lip_model = init(wav2lip_checkpoint_path, device)

    # load tmp data
    video_path = './sample_data/input_vid_portrait_orig.m4v'
    images_cv, video_fps = api.extract_video_frames(video_path)
    audio_path = "./sample_data/input_audio.wav"
    audio_pydub = AudioSegment.from_wav(audio_path)

    # predictor
    with Timer(name="Processing", text="{name}: Elapsed time: {milliseconds:.0f} ms"):
        processed_frames = predict(face_alignment_detector, wav2lip_model, images_cv, audio_pydub, video_fps,
                               device=device,
                               face_det_batch_size=10,
                               wav2lip_batch_size=10)


    # write video to disk
    outfile = 'temp/final.avi'
    frame_h, frame_w = processed_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), video_fps, (frame_w, frame_h))
    for img in processed_frames:
        out.write(img)
    out.release()

    # write the audio segment to disk


    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', outfile)
    subprocess.call(command, shell=True)
