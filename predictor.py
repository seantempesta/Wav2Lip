import subprocess
import cv2
import face_alignment
import torch
import json
from pydub import AudioSegment

from models import Wav2Lip
import predictor_api as api
# CONSTANTS
AUDIO_FRAME_RATE = 16000


def init(wav2lip_checkpoint_path, device="cuda"):
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


# function to generate speech
def predict(face_alignment_detector, wav2lip_model, images_cv, audio_pydub, args, device="cuda"):



    return True

def repl_test():
    wav2lip_checkpoint_path = './wav2lip_gan.pth'
    device = "cuda"
    mouth_mask_path = './mouth_mask.npy'

    # init
    face_alignment_detector, wav2lip_model = init(wav2lip_checkpoint_path,device)

    # load tmp data
    argv = ["--wav2lip_checkpoint_path", "wav2lip_gan.pth", "--face",
            "./sample_data/input_vid_portrait_orig.m4v", "--audio", "./sample_data/input_audio.wav"]
    args = api.gen_args(argv)
    args.pads = [10,10,0,0]

    args.face_det_confidence_score_min = 0.8

    video_path = './sample_data/input_vid_portrait_orig.m4v'
    images_cv, orig_video_fps = api.extract_video_frames(video_path)
    args.fps = orig_video_fps
    audio_path = "./sample_data/input_audio.wav"
    audio_pydub = AudioSegment.from_wav(audio_path)


    # predictor
    data = api.preprocess_data(face_alignment_detector, images_cv, audio_pydub, args)
    predictions = api.process_in_batches(data, 10,
                                     lambda x: api.process_wav2lip_batch(x, wav2lip_model, args, device="cuda"))
    final_frames = api.postprocess_wav2lip(images_cv, predictions, data, mouth_mask_path)

    frame_h, frame_w = final_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), args.fps, (frame_w, frame_h))

    for img in final_frames:
        out.write(img)
    out.release()

    # write the audio segment to disk


    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=True)
