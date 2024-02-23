import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    dataset_path = "/home/alex/DATASETS/TODO/HuTics"
    split_path = "/home/alex/DATASETS/TODO/HuTics/train_test_split.json"
    im_folder = "img"
    hands_folder = "hand"
    obj_folder = "objmask"
    batch_size = 30

    def create_ann(image_path):
        labels = []
        tags = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        seq_value = image_path.split("/")[-2]
        tag = sly.Tag(sequence_meta, value=seq_value)
        tags.append(tag)

        labels_tag_val = get_file_name(image_path).split("-")[0]
        l_meta = meta.get_tag_meta(labels_tag_val)
        l_tag = sly.Tag(l_meta)

        hand_path = image_path.replace(im_folder, hands_folder)

        if file_exists(hand_path):
            mask_np = sly.imaging.image.read(hand_path)[:, :, 0]
            mask = mask_np == 255
            if len(np.unique(mask)) > 1:
                ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
                for i in range(1, ret):
                    obj_mask = curr_mask == i
                    curr_bitmap = sly.Bitmap(obj_mask)
                    if curr_bitmap.area > 25:
                        curr_label = sly.Label(curr_bitmap, hand, tags=[l_tag])
                        labels.append(curr_label)

        obj_path = image_path.replace(im_folder, obj_folder)

        if file_exists(obj_path):
            mask_np = sly.imaging.image.read(obj_path)[:, :, 0]
            qwe = np.unique(mask_np)
            mask = mask_np == 255
            if len(np.unique(mask)) > 1:
                curr_bitmap = sly.Bitmap(mask)
                curr_label = sly.Label(curr_bitmap, obj, tags=[l_tag])
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    hand = sly.ObjClass("hand", sly.Bitmap)
    obj = sly.ObjClass("object of interest", sly.Bitmap)

    sequence_meta = sly.TagMeta("sequence", sly.TagValueType.ANY_STRING)
    pointing_meta = sly.TagMeta("pointing", sly.TagValueType.NONE)
    present_meta = sly.TagMeta("present", sly.TagValueType.NONE)
    touch_meta = sly.TagMeta("touch", sly.TagValueType.NONE)
    exhibit_meta = sly.TagMeta("exhibit", sly.TagValueType.NONE)

    meta = sly.ProjectMeta(
        obj_classes=[hand, obj],
        tag_metas=[sequence_meta, pointing_meta, present_meta, touch_meta, exhibit_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    idx_to_name = {}
    split_data = load_json_file(split_path)

    for ds_name, folders in split_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        for folder in folders:
            images_path = os.path.join(dataset_path, folder, im_folder)

            images_names = os.listdir(images_path)

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                new_im_names = []
                img_pathes_batch = []
                for im_name in images_names_batch:
                    img_pathes_batch.append(os.path.join(images_path, im_name))
                    new_im_names.append(folder + "_" + im_name)

                img_infos = api.image.upload_paths(dataset.id, new_im_names, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(images_names_batch))

    return project
