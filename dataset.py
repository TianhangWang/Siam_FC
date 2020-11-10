from torch.utils.data.dataset import Dataset
import json
from Utils import *

class ILSVRCDataSet(Dataset):
    def __init__(self, lmdb, data_dir, config, z_transforms,
                                    x_transforms, mode='Train'):
        lmdb_video = json(load(open(imdb, 'r')))
        self.videos = lmdb_video['videos']
        self.data_dir = data
        self.config = config
        # how json file work? perheps can access the item by attr.
        self.num_videos = int(imdb_video['num_videos'])

        self.z_transforms = z_transforms
        self.x_transforms = x_transforms

        # question? why do that?
        if mode == "Train":
            self.num = self.config.num_pairs
        else:
            self.num = self.num_videos
        
    def __getitem__(self, rand_vid):
        """
        read a pair of images z and x
        """
        rand_vid = rand_vid % self.num_videos

        videos_keys = self.videos.keys()
        video = self.videos[videos_keys[rand_vid]]
        video_ids = video[0]
        video_id_keys = video_id_keys.keys()

        rand_trackid_z = np.random.choice(list(range(len(video_id_keys))))
        video_id_z = video_ids[video_id_keys[rand_trackid_z]]

        rand_z = np.random.choice(range(len(video_id_z)))

        # pick a valid instance within frame_range frames from the examplar, excluding the examplar itself
        possible_x_pos = range(len(video_id_z))
        rand_x = np.random.choice(possible_x_pos[max(rand_z - self.config.pos_pair_range, 0):rand_z] + possible_x_pos[(rand_z + 1):min(rand_z + self.config.pos_pair_range, len(video_id_z))])

        z = video_id_z[rand_z].copy()    # use copy() here to avoid changing dictionary
        x = video_id_z[rand_x].copy()

        # read z and x
        img_z = cv2.imread(os.path.join(self.data_dir, z['instance_path']))
        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)

        img_x = cv2.imread(os.path.join(self.data_dir, x['instance_path']))
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        # do data augmentation;
        # note that we have done center crop for z in the data augmentation
        img_z = self.z_transforms(img_z)
        img_x = self.x_transforms(img_x)

        return img_z, img_x

    def __len__(self):
        return self.num

