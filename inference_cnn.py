import cv2
import os
import mediapipe as mp
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import model as m
import time
import matplotlib.pyplot as plt
import yaml
from argparse import ArgumentParser

from retinaface_detect import detect as retinaface_detect_faces
from retinaface_detect import create_net as retinaface_model

from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework


def process_frame(image, model, mp_face_mesh):
    normal_feature = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    try:
        features = model.process(image).multi_face_landmarks[0].landmark
    except Exception as _:
        features = []
        normal_feature = False
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape
    landmarks = [[f.x * w, f.y * h] for f in features]

    # inverted view, mediapipe uses view from eyes
    left_indices = mp_face_mesh.FACEMESH_RIGHT_EYE
    right_indices = mp_face_mesh.FACEMESH_LEFT_EYE

    if normal_feature:
        raw_left_landmarks = [landmarks[idx[0]] for idx in left_indices]
        raw_right_landmarks = [landmarks[idx[0]] for idx in right_indices]

        # idx - pos
        # 1 - 0, 3 - 2, 4 - 7, 6 - 3, 8 - 5, 13 - 1, 14 - 6, 15 - 4
        lidx = [1, 3, 15, 14]
        left_landmarks = []
        for idx in lidx:
            left_landmarks.append(raw_left_landmarks[idx])

        # idx - pos
        # 0 - 6, 3 - 1, 4 - 5, 5 - 4, 6 - 3, 10 - 2, 13 - 7, 14 - 0
        ridx = [14, 10, 5, 0]
        right_landmarks = []
        for idx in ridx:
            right_landmarks.append(raw_right_landmarks[idx])

        left_landmarks = np.array(left_landmarks)
        right_landmarks = np.array(right_landmarks)

        return left_landmarks, right_landmarks
    else:
        return [], []


def process_frame_spiga(frame, processor, retinaface_net, retinaface_cfg):
    faces = retinaface_detect_faces(frame, retinaface_net, retinaface_cfg)

    normal_feature = True
    try:
        bbox = extract_bboxes(faces, 0.99)[0]
        features = processor.inference(frame, [bbox])
        left_land = np.array(features['landmarks'][0])[60:68]
        right_land = np.array(features['landmarks'][0])[68:76]
    except Exception as _:
        features = []
        normal_feature = False

    if not normal_feature:
        left_land, right_land = [], []

    return left_land, right_land


def xyxy_to_xywh(bbox):
    new_bbox = [0.0] * len(bbox)

    new_bbox[0] = bbox[0]
    new_bbox[1] = bbox[1]
    new_bbox[2] = bbox[2] - bbox[0]
    new_bbox[3] = bbox[3] - bbox[1]

    return new_bbox


def extract_bboxes(faces, threshold):
    bboxes = []
    for face in faces:
        if face[4] < threshold:
            continue

        bbox = face[:4]
        new_bbox = xyxy_to_xywh(bbox)

        bboxes.append(new_bbox)
    return bboxes


def get_bbox(land, pad):
    hpad, vpad = pad
    l, r, t, b = land[0][0], land[2][0], land[1][1], land[3][1]

    l = int(l - hpad)
    r = int(r + hpad)
    t = int(t - vpad)
    b = int(b + vpad)

    return l, r, t, b


def display_values(image,
                   fps,
                   freq,
                   dur,
                   aecd,
                   font_size=0.4,
                   thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX

    str_fps = f'FPS: {fps:.2f}'
    str_freq = f'FREQ: {freq:.3f} b/s'
    str_dur = f'DUR: {dur:.3f} s'
    str_aecd = f'AECD: {aecd:.3f} s/b'

    cv2.putText(image,
                str_fps,
                (10, 20),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    cv2.putText(image,
                str_freq,
                (10, 40),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    cv2.putText(image,
                str_dur,
                (10, 60),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    cv2.putText(image,
                str_aecd,
                (10, 80),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    return image


def plot_graphics(name,
                  frequencies,
                  durations,
                  aecds,
                  fps=60.0,
                  grid=True,
                  num_xticks=10,
                  num_yticks=10):
    num = len(frequencies)
    sec = num / fps

    x = np.linspace(0.0, sec, num)
    x_max = x[-1]

    freq_max = np.max(frequencies)
    dur_max = np.max(durations)
    aecd_max = np.max(aecds)

    x_ticks_val = np.linspace(0.0, x_max, num=num_xticks)
    x_ticks_label = [f'{el:.2f}' for el in x_ticks_val]

    y_ticks_freq_val = np.linspace(0.0, freq_max, num=num_yticks)
    y_ticks_freq_label = [f'{el:.2f}' for el in y_ticks_freq_val]

    y_ticks_dur_val = np.linspace(0.0, dur_max, num=num_yticks)
    y_ticks_dur_label = [f'{el:.2f}' for el in y_ticks_dur_val]

    y_ticks_aecd_val = np.linspace(0.0, aecd_max, num=num_yticks)
    y_ticks_aecd_label = [f'{el:.2f}' for el in y_ticks_aecd_val]

    _, (ax11, ax12, ax13) = plt.subplots(1,
                                         3,
                                         figsize=(10, 5),
                                         layout='constrained')

    ax11.set_title(f'FREQ')
    ax11.set_xlim(0, x_max)
    ax11.set_ylim(0, freq_max)
    ax11.grid(grid)
    ax11.plot(x, frequencies)
    ax11.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax11.set_yticks(y_ticks_freq_val, y_ticks_freq_label)

    ax12.set_title(f'DUR')
    ax12.set_xlim(0, x_max)
    ax12.set_ylim(0, dur_max)
    ax12.grid(grid)
    ax12.plot(x, durations)
    ax12.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax12.set_yticks(y_ticks_dur_val, y_ticks_dur_label)

    ax13.set_title(f'AECD')
    ax13.set_xlim(0, x_max)
    ax13.set_ylim(0, aecd_max)
    ax13.grid(grid)
    ax13.plot(x, aecds)
    ax13.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax13.set_yticks(y_ticks_aecd_val, y_ticks_aecd_label)

    plt.savefig(name)


def parse():
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='./config_cnn.yaml')
    args = parser.parse_args()
    return args


def parse_yaml(file):
    with open(file) as f:
        my_dict = yaml.safe_load(f)
    return my_dict


def main():
    args = parse()
    config = parse_yaml(args.config)

    prefix = config['prefix']
    init_fps = config['fps']
    model_name = config['model_name']
    retinaface_backbone = config['retinaface_backbone']
    video_path = prefix + config['input_video']
    res_video = prefix + f'/{model_name}_{config["output_video"]}'
    graphics = prefix + f'/{model_name}_{config["graphics"]}'
    use_cpu = config['use_cpu']
    dataset = 'wflw'
    pad = (25.0, 50.0)

    mp_face_mesh = mp.solutions.face_mesh
    face_model = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    retinaface_weights = './retinaface_pytorch/weights/'
    if retinaface_backbone == 'resnet50':
        retinaface_weights += 'Resnet50_Final.pth'
    elif retinaface_backbone == 'mobile0.25':
        retinaface_weights += 'mobilenet0.25_Final.pth'

    ret_net, ret_cfg = retinaface_model(network=retinaface_backbone,
                                        weights=retinaface_weights)

    spiga_processor = SPIGAFramework(ModelConfig(dataset), use_cpu=use_cpu)

    model = getattr(m, 'ResNet20')(3, 2)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    best_ckpt = os.path.join('./ckpt', 'best.pth')

    state_dict = torch.load(best_ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((50, 50))
    ])

    prev_frame_time = 0
    new_frame_time = 0
    blink_frames = 0
    blink_cnt = 0
    blink_cont = False
    idx = 0

    durs = []
    freqs = []
    aecds = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    mean_fps = 0.0

    video_cap = cv2.VideoCapture(video_path)
    while (video_cap.isOpened()):
        ret, frame = video_cap.read()
        if ret:
            if writer is None:
                h, w, _ = frame.shape
                writer = cv2.VideoWriter(res_video, fourcc, init_fps, (w, h))

            idx += 1
            prev_frame_time = time.time()

            if model_name == 'spiga':
                left_land, right_land = process_frame_spiga(frame,
                                                            spiga_processor,
                                                            ret_net,
                                                            ret_cfg)
            else:
                left_land, right_land = process_frame(frame,
                                                      face_model,
                                                      mp_face_mesh)

            if len(left_land) > 0 and len(right_land) > 0:
                ll, lr, lt, lb = get_bbox(left_land, pad)
                rl, rr, rt, rb = get_bbox(right_land, pad)

                left_eye = frame[lt:lb + 1, ll:lr + 1, :]
                right_eye = frame[rt:rb + 1, rl:rr + 1, :]

                left_eye = transform(left_eye).to(device).unsqueeze(0)
                right_eye = transform(right_eye).to(device).unsqueeze(0)

                outputs = model(left_eye, right_eye)
                _, predicted = torch.max(outputs.data, 1)

                if predicted.item() == 1:
                    if not blink_cont:
                        blink_cont = True
                        blink_cnt += 1
                    blink_frames += 1

                    overlay = frame.copy()
                    h, w, _ = frame.shape

                    start_point = (int(0), int(0))
                    end_point = (int(w), int(h))

                    cv2.rectangle(overlay,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=(0, 200, 0),
                                  thickness=-1)
                    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                else:
                    blink_cont = False

                dur = blink_frames / init_fps
                freq = blink_cnt / idx * init_fps
                aecd = dur / blink_cnt if blink_cnt != 0 else 0.0
            else:
                dur = 0.0
                freq = 0.5
                aecd = 0.0

            durs.append(dur)
            freqs.append(freq)
            aecds.append(aecd)

            new_frame_time = time.time()
            fps = 1.0 / (new_frame_time - prev_frame_time)

            mean_fps += fps

            frame = display_values(frame, fps, freq, dur, aecd)
            writer.write(frame)

            print(f'processed {idx} frames')
        else:
            break

    print(f'fps {(mean_fps / idx):.2f}')

    mean_fqr = f'{sum(freqs) / len(freqs):.3f}'
    print('mean frequency', mean_fqr)
    print(f'last frequnecy {freqs[-1]:.3f}')
    print(f'duration  {durs[-1]:.3f}')
    print(f'aecd {aecds[-1]:.3f}')

    plot_graphics(graphics,
                  freqs,
                  durs,
                  aecds,
                  init_fps)

    video_cap.release()
    writer.release()


if __name__ == '__main__':
    main()
