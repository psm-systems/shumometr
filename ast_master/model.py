import csv
import os
import wget
import numpy as np
import torch
import torchaudio
import pandas as pd
import plotly.express as px
from torch.amp import autocast
from src.models import ASTModel


# Create a new class that inherits the original ASTModel class
class ASTModelVis(ASTModel):
    @staticmethod
    def get_att_map(block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        b, n, c = x.shape
        qkv = qkv(x).reshape(b, n, 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
        q, qkv1, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ qkv1.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        b = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(b, -1, -1)
        dist_token = self.v.dist_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list


def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    #    assert sr == 16000, 'input audio sampling rate must be 16kHz'
    # , sample_frequency=sr
    fbank = torchaudio.compliance.kaldi.fbank(waveform,
                                              htk_compat=True,
                                              use_energy=False,
                                              window_type='hanning',
                                              num_mel_bins=mel_bins,
                                              dither=0.0,
                                              frame_shift=10)
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(labels_csv):
    with open(labels_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels_lst = []
    ids = []  # Each label has a unique identificator such as "/m/068hy"
    for i1 in range(1, len(lines)):
        identificator = lines[i1][1]
        label = lines[i1][2]
        ids.append(identificator)
        labels_lst.append(label)
    return labels_lst


def ast_model_main(path, name):
    # Create an AST model and download the AudioSet pretrained weights
    audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
    if os.path.exists('pretrained_models/audio_mdl.pth') is False:
        wget.download(audioset_mdl_url, out='./pretrained_models/audio_mdl.pth')

    # Assume each input spectrogram has 1024 time frames
    input_tdim = 1024
    checkpoint_path = './pretrained_models/audio_mdl.pth'

    # now load the visualization model
    ast_mdl = ASTModelVis(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cpu:0"))
    audio_model.eval()

    # Load the AudioSet label set
    label_csv = '../ast_master/egs/audioset/data/class_labels_indices.csv'
    labels = load_label(label_csv)

    feats = make_features(f'{path}{name}.wav', mel_bins=128)
    feats_data = feats.expand(1, input_tdim, 128)
    feats_data = feats_data.to(torch.device("cpu:0"))

    # do some masking of the input
    # feats_data[:, :512, :] = 0

    # Make the prediction
    with torch.no_grad():
        with autocast('cuda'):
            output = audio_model.forward(feats_data)
            output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]

    ans = []
    nums = []

    for k in range(5):
        ans.append(np.array(labels)[sorted_indexes[k]])
        nums.append(result_output[sorted_indexes[k]] * 100)

    data = pd.DataFrame({'Names': ans, 'Values': nums})

    # Список используемых цветов
    color = ['#636EFA',
 '#EF553B',
 '#00CC96',
 '#AB63FA',
 '#FFA15A']

    fig = px.bar(
        data_frame=data,  # датафрейм
        y="Names",
        x="Values",
        labels={'Names': '', 'Values': 'Sounds, %'},
        color=color[0:5],  # расцветка в зависимости от value
        text='Names',  # текст на столбцах
        orientation='h',  # ориентация графика
        height=500,  # высота
        width=1000,  # ширина
    )
    fig.update_traces(textfont_size=50, textangle=0, textposition="auto") # Подпись столбцов
    fig.update_layout(showlegend=False, title_x=0.5,  title_y=0.95,
       font=dict(
           family="Arial, sans-serif",
           size=50,
           color="Black"
       )
   )
    fig.update_yaxes(tickvals=[], title_text='')

    # Запись изображения в файл
    fig.write_image(f'../image_message/{name}.png', width=1920, height=1080,  engine="kaleido", validate=True)

    res_message = 'Я проанализировал твое голосовое сообщение 🔊 : \n\n'
    for i in range(5):
        res_message += f'{ans[i]} - {round(float(nums[i]), 2)} %\n'

    return res_message


