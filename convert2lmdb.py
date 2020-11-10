import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import json

validation_ratio = 0.1

def generate_img_imdb(vid_root_path, vid_curated_path):

    anno_str = "Annotations/VID/train/"
    data_str = "Data/VID/train/"
    vid_anno_path = os.path.join(vid_root_path, anno_str)
    vid_data_path = os.path.join(vid_root_path, data_str)

    num_videos = 0

    all_dirs_level_one = os.listdir(vid_anno_path)
    
    for i in range(len(all_dirs_level_one)):
        all_dirs_level_two = os.listdir(os.path.join(vid_anno_path, all_dirs_level_one[i]))
        num_videos = num_videos + len(all_dirs_level_two)
    
    train_video_num = round(num_videos * (1 - validation_ratio))
    val_video_num = num_videos - train_video_num

    imdb_video_train = dict()
    imdb_video_train['num_video'] = train_video_num
    imdb_video_train['data_str'] = data_str

    imdb_video_val = dict()
    imdb_video_val['num_video'] = val_video_num
    imdb_video_val['data_str'] = data_str

    videos_train = dict()
    videos_val = dict()

    vid_idx = 0

    for i in range(len(all_dirs_level_one)):
        all_dirs_level_two = os.listdir(os.path.join(vid_anno_path, all_dirs_level1[i]))
        # train_set's unit is each video.
        for j in range(len(all_dirs_level_two)):
            # 它的抽取是按照文件名顺序挨个抽取的
            if vid_idx < train_video_num:
                if not videos_train.has_key(all_dirs_level_two[j]):
                    videos_train[all_dirs_level_two[j]] = []
            else:
                if not videos_val.has_key(all_dirs_level_two[j]):
                    videos_val[all_dirs_level_two[j]] = []
            
            frame_list = glob.glob(os.path.join(vid_anno_path, all_dirs_level_one[i], all_dirs_level_two[j],"*.xml"))
            frame_list.sort()

            video_ids = dict()
            # from a single video to process each frame.
            # train set -> video -> frame
            # in frame sequentials, we sort frames by object id
            """
            ----train_set(dict)
                ----video(dict)
                    ----object id(dict)
                        ---- path, bbox
            """
            for k in range(len(frame_list)):
                frame_id = k
                frame_xml_name = os.path.join(vid_anno_path, all_dirs_level1[i], all_dirs_level2[j], frame_list[k])
                frame_xml_tree = ET.parse(frame_xml_name)
                frame_xml_root = frame_xml_tree.getroot()  

                crop_path = os.path.join(all_dirs_level_one[i], all_dirs_level_two[j])
                frame_filename = frame_xml_root.find('filename').text

                print("processing: %s, %s, %s ...") % (all_dirs_level_one[i], all_dirs_level_two[j], frame_xml_name)
                for object in frame_xml_root.iter("object"):
                    id = object.find("trackid").text
                    if not video_ids.has_key(id):
                        video_ids[id] = []

                    bbox_node = object.find("bndbox")
                    xmax = float(bbox_node.find('xmax').text)
                    xmin = float(bbox_node.find('xmin').text)
                    ymax = float(bbox_node.find('ymax').text)
                    ymin = float(bbox_node.find('ymin').text)
                    width = xmax - xmin + 1
                    height = ymax - ymin + 1
                    bbox = np.array([xmin, ymin, width, height])

                    tmp_instance = dict()
                    tmp_instance['instance_path'] = os.path.join(all_dirs_level_one[i], all_dirs_level_two[j],'{}.{:02d}.crop.x.jpg'.format(frame_filename, int(id)))
                    tmp_instance['bbox'] = bbox.tolist()

                    video_ids[id].append(tmp_instance)
            # before analysis next video, filter current video
            tmp_keys = video_ids.keys()
            for ki in range(len(tmp_keys)):
                if len(video_ids[tmp_keys[ki]]) < 2:
                    del video_ids[tmp_keys[ki]]
            
            tmp_keys = video_ids.keys()

            if len(tmp_keys) > 0:
                if vid_idx < train_video_num:
                    # sorted by vedio folder. 
                    videos_train[all_dirs_level_two[j]].append(video_ids)
                else:
                    videos_val[all_dirs_level_two[j]].append(video_ids)

                vid_idx+=1
    imdb_video_train['video'] = videos_train
    imdb_video_val['video'] = videos_val

    json.dump(imdb_video_train, open('imdb_video_train.json','w'), indent=2)
    json.dump(imdb_video_val, open('imdb_video_val.json', 'w'), indent=2)

if __name__ == "__main__":
    vid_root_path = ''
    vid_curated_path = ''
    generate_img_imdb(vid_root_path, vid_curated_path)

                
                
