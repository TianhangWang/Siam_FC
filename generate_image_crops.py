import numpy as np
import os 
import glob
import xml.etree.ElementTree as ET
import cv2
import datetime

examplar_size = 127.0
instance_size = 255.0
context_amount = 0.5

def get_subwindow_avg(im, pos, model_sz, original_sz):
    """
    to fix the image.
    """
    # pos -> center position.
    avg_chans = [np.mean(im[:,:,0]), np.mean(im[:,:,1],), np.mean(im[:,:,2])]
    if original_sz is None:
        original_sz = model_sz

    # 
    sz = original_sz
    # get the shape of image
    im_sz = im.shape
    assert (im_sz[0] > 2) && (im_sz[1] > 2), "The size of image is too small"
    # get the lenght of half edge. why add 1? cause is odd.
    c = (sz + 1) / 2
    # get round(), cause the pix is int.
    context_xmin = round(pos[1] - c)
    # cause python: width = max - min + 1
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[0] - c)
    context_ymax = context_ymin + sz - 1
    
    # if context_xmin is out of boundry, you get negetive value
    # the left_pad will not be zero! Using this to determine whether
    # to add padding!
    left_pad = max(0, 1 - context_xmin)
    top_pad = max(0, 1 - context_ymin)
    right_pad = max(0, context_xmax - im_sz[1])
    bottom_pad = max(0, context_ymax - im_sz[0])

    # here I think author got some problems.
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + right_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + bottom_pad

    im_R = im[:,:,0]
    im_G = im[:,:,1]
    im_B = im[:,:,2]

    # padding
    # see more https://blog.csdn.net/zenghaitao0128/article/details/78713663
    if ((top_pad != 0) | (bottom_pad != 0) | (left_pad != 0) | (right_pad != 0)):
        im_R = np.pad(im_R,((int(top_pad), int(bottom_pad)), (int(left_pad, int(right_pad)))),
                                constant-values = avg_chans[0])
        im_G = np.pad(im_G,((int(top_pad), int(bottom_pad)), (int(left_pad, int(right_pad)))),
                                constant-values = avg_chans[1])
        im_B = np.pad(im_B,((int(top_pad), int(bottom_pad)), (int(left_pad, int(right_pad)))),
                                constant-values = avg_chans[2]])

        im = np.stack((im_R,im_G,im_B), axis=2)
    # Now, the image has been processed, already take boundry into accont.
    
    im_patch_original = im[int(context_ymin)-1:int(context_ymax),int(context_xmin)-1:int(context_xmax),:]
    
    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch_original, (int(model_sz),int(mode_sz)), interpolation=cv2.INTER_CUBIC)
    else:
        im_patch = im_patch_original
    
    return im_patch

def get_crops(img, bbox, size_z, size_x, context_amount):

    """
    get examplar and search region crops
    bbox -> you get from .xml file.
    """
    
    center_x = bbox[0] + bbox[2] / 2 
    center_y = bbox[1] + bbox[3] / 2
    w = bbox[2]
    h = bbox[3]
    # By paper, the author design the context margin.
    # for examplar
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z + hc_z)
    scale_z = size_z / s_z
    im_crop_z = get_subwindow_avg(img, np.arr([cy,cx]), size_z, round(s_z))

    # for search 
    # here, I still got some questions
    # I will debug the code using real data to see how it operate.
    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad 
    scale_x = size_x / s_x 
    im_crop_x = get_subwindow_avg(img, np.array([cy, cx], size_x, round(s_x))

    return im_crop_z, im_crop_x

def generate_image_crops(vid_root_path, vid_curated_path):
    anno_str = "Annotations/VID/train/"
    data_str = "Data/VID/train/"

    vid_anno_path = os.path.join(vid_anno_path, anno_str)
    vid_data_path = os.path.join(vid_curated_path, data_str)

    cur_processed_frame = 0
    start_time = datetime.datetime.now()
    total_time = 0

    all_dirs_lever_one = os.listdir(vid_anno_path)
    for i in range(len(all_dirs_lever_one)):

        all_dirs_lever_two = os.listdir(os.path.join(vid_anno_path, all_dirs_lever_one[i]))
        
        for j in range(len(all_dirs_lever_two)):
            frame_list = glob.glob(os.path.join(vid_anno_path,all_dirs_lever_one[i],all_dirs_lever_two[j], "*.xml"))
            frame_list.sort()

            for k in range(len(frame_list)):
                frame_xml_name = os.path.join(vid_anno_path,all_dirs_lever_one[i],all_dirs_lever_two[j],frame_list[k])
                frame_xml_tree = ET.parse(frame_xml_name)
                frame_xml_root = frame_xml_tree.getroot()

                frame_img_name = (frame_list[k].replace(".xml",".JPEG")).replace(vid_anno_path, vid_data_path)
                img = cv2.imread(frame_img_name)
                if img is None:
                    print("Cannot find {}".format(frame_img_name))
                    exit(0)
                frame_filename = frame_xml_root.find('filename').text

                for object in frame_xml_root.iter("object"):
                    id = object.find("trackid").text
                    bbox_node = object.find("bndbox")
                    xmax = float(bbox_node.find('xmax')).text
                    xmin = float(bbox_node.find('xmin')).text
                    ymax = float(bbox_node.find('ymax')).text
                    ymin = float(bbox_node.find('ymin')).text
                    width = xmax - xmin + 1
                    height = ymax - ymin + 1
                    bbox = np.array([xmin,ymin,width,height])
                    im_crop_z, im_crop_x = get_crops(img, bbox, examplar_size, instance_size, context_amount)
                    save_path = os.path.join(vid_curated_path, data_str, all_dirs_level1[i], all_dirs_level2[j])
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # it seem that the exam
                    savename_crop_z = os.path.join(save_path, '{}.{:02d}.crop.z.jpg'.format(frame_filename,int(id)))
                    savename_crop_x = os.path.join(save_path, '{}.{:02d}.crop.x.jpg'.format(frame_img_name,int(id)))

                    cv2.imwrite(savename_crop_z, im_crop_z, [int(cv2.IMWRITE_JPEG_QUALITY),90])
                    cv2.imwrite(savename_crop_x, im_crop_x, [int(cv2.IMWRITE_JPEG_QUALITY),90])

                    cur_processed_frame +=1
                    if cur_processed_frame % 1000 == 0:
                        end_time = datetime.datetime.now()
                        total_time = total_time + int((end_time - start_time).seconds)
                        print("finished processing %d frames in %d seconds (FPS: %d) ..." % (cur_processed_frame, total_time, int(100/(end_time-start_time).seconds)))
                        start_time = datetime.datetime.now()

if __name__ == "__main__":
    vid_root_path = ''
    vid_curated_path = ''
    if not os.path.exists(vid_curated_path):
        os.makedirs(vid_curated_path)
    generate_image_crops(vid_root_path, vid_curated_path)  

