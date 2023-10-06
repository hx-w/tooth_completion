# -*- coding: utf-8 -*-

import os
import hashlib

import gradio as gr
import plotly.express as px
import pandas as pd

from preprocess import sample_sdf_npz
from inference import reconstruct_latent, extract_mesh_from_latent, compute_mesh_errors


def hash_file(filename):
    with open(filename, 'rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()

def entry_defect_mesh(mesh_path, location, iters, resols, progress=gr.Progress(track_tqdm=True)):
    # check location model exist
    exp_model = os.path.join('experiments', f'{location[1:]}_Outside')
    if not os.path.isfile(f'{exp_model}/ModelParameters/latest.pth'):
        raise gr.Error(f'神经网络模型 {location[1:]}_Outside 不存在！')

    ## Preprocess
    progress(0, desc="开始采样")
    uuid = hash_file(mesh_path)
    npz_name = os.path.join('.cache', f'{uuid}.npz')
    if not os.path.isfile(npz_name):
        sample_sdf_npz(mesh_path, npz_name)
    else:
        gr.Info('采样结果已缓存，跳过采样过程')

    ## Inference
    resols = {'低 (128x128)': 128, '中 (256x256)': 256, '高 (512x512)': 512}[resols]
    progress(0, desc="开始重建符号距离场")
    err, latent, decoder = reconstruct_latent(npz_name, iters, exp_model)
    gr.Info(f'符号距离场重建完成，误差为 {err:.4f}')

    ## Extract Mesh
    target_mesh = os.path.join('.cache', f'{uuid}')
    progress(0, desc="开始提取网格")
    sdf_slices = extract_mesh_from_latent(decoder, latent, target_mesh, resols)

    progress(0, desc="开始计算误差分布")
    x_axis, y_axis = compute_mesh_errors(target_mesh + '.obj', mesh_path)
    df = pd.DataFrame({'ranges': x_axis, 'counts': y_axis})
    fig = px.line(df, x='ranges', y='counts')
    fig.update_layout(
        title="误差分布图",
        xaxis_title="误差区间",
        yaxis_title="计数",
    )

    # if os.path.isfile(npz_name):
    #     os.remove(npz_name)

    return target_mesh + '.obj', sdf_slices, fig

def entry_template_mesh(location):
    template_mesh = os.path.join('templates', f'{location[1:]}_Outside.obj')
    if not os.path.isfile(template_mesh):
        raise gr.Error(f'模板牙齿模型 {location[1:]}_Outside 不存在')

    return template_mesh


if __name__ == "__main__":
    if not os.path.exists('.cache'):
        os.mkdir('.cache')

    with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
        gr.Markdown(
            '# 残缺牙齿重建\n'
            '本项目使用神经网络重建残缺牙齿的三维模型 **ToothDIT** 实现。\n'
            '从最左侧上传残缺牙齿模型，选择牙齿位置，点击开始重建即可。\n<br/>'
            '重建过程包括：`网格采样` -> `符号距离场重建` -> `网格提取` -> `误差分布计算`。'
        )
        with gr.Row(equal_height=True):
            with gr.Column():
                inp_mesh = gr.Model3D(label="残缺牙齿模型")
                inp_loc = gr.Radio(
                    ["#11", "#12", "#21", "#22", "#15"],
                    value="#11",
                    label="牙齿位置",
                )
                inp_iters = gr.Slider(300, 1500, step=10, label="迭代次数", value=800)
                inp_resols = gr.Dropdown(
                    ["低 (128x128)", "中 (256x256)", "高 (512x512)"],
                    value="中 (256x256)",
                    label="网格重建精度", info="括号中为Marching Cubes的体素精度"
                )

            with gr.Column():
                otp_mesh = gr.Model3D(label="完整牙齿模型")
                otp_template = gr.Model3D(label="模板牙齿模型", value=entry_template_mesh('#11'))
            
            with gr.Column(visible=False) as trd_col:
                otp_slices = gr.Gallery(
                    label="完整牙齿SDF切面", show_label=True, elem_id="gallery",
                    columns=[3], rows=[1], object_fit="fill", height='500'
                )
                otp_errors = gr.Plot(
                    label="逐点误差图", show_label=True
                )

        inp_loc.change(entry_template_mesh, [inp_loc], outputs=[otp_template])

        def submit(inp_mesh, inp_loc, inp_iters, inp_resols, progress=gr.Progress(track_tqdm=True)):
            xotp_mesh, xotp_slices, xotp_fig = entry_defect_mesh(inp_mesh, inp_loc, inp_iters, inp_resols, progress)
            return {
                trd_col: gr.update(visible=True),
                otp_mesh: xotp_mesh,
                otp_slices: xotp_slices,
                otp_errors: xotp_fig
            }

        btn_commit = gr.Button("开始重建")
        btn_commit.click(
            submit,
            inputs=[inp_mesh, inp_loc, inp_iters, inp_resols],
            outputs=[otp_mesh, otp_slices, otp_errors, trd_col]
        )
        

    demo.queue().launch(share=True)
