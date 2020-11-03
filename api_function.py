import argparse


import glob
import os
import sys
#dataset preprocessing functions
from cloth_mask import grabcut
sys.path.insert(1, os.path.join(sys.path[0] ,"keypoints"))
from keypoints.get_output import *
from test_pgn import main as segment_images
from DatasetPreparationSemanticParsing import semantic_dataloader
from PIL import Image
from swap_out import *

IMAGE_WIDTH = 192
IMAGE_HEIGHT = 256

def get_keypoints_parser(paths):
    parser = argparse.ArgumentParser(
        description='''Cloth Swap model API''')
    parser.add_argument('--checkpoint-path', type=str, default = os.path.join(sys.path[0], "keypoints/checkpoints/checkpoint_iter_370000.pth"), help='path to the keypoints checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--images_dir', type = str, default= paths["image"], help='path to input images directory')
    parser.add_argument('--output_dir', type = str, default=paths["key_pose"], help='path to output images directory')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')

    return parser.parse_args()

def get_opt(paths,name, stage, checkpoint ):

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default=name)
    # parser.add_argument("--name", default="TOM")

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default=os.path.dirname(paths["base"]))

    # parser.add_argument("--datamode", default="train")
    parser.add_argument("--datamode", default="test")

    parser.add_argument("--stage", default=stage)
    # parser.add_argument("--stage", default="TOM")

    # parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--data_list", default="test_pairs.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    if stage == "TOM":
        parser.add_argument('--result_dir', type=str,
                        default=os.path.join(os.path.dirname(paths["base"]), "result" ), help='save result infos')
    else:
        parser.add_argument('--result_dir', type=str,
                        default=paths["base"], help='save result infos')

    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='model checkpoint for test')
    # parser.add_argument('--checkpoint', type=str, default='checkpoints/TOM/tom_final.pth', help='model checkpoint for test')

    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt

def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def create_keypoints(paths):
    
    args = get_keypoints_parser(paths)
    if  args.images_dir == '':
        raise ValueError('--image_directory  has to be provided')



    if args.images_dir:
        image_paths = glob.glob(args.images_dir + "/*.*g")
        frame_provider = ImageReader(image_paths)
    else:
        frame_provider = ImageReader(args.images)
    args.track = 0

    run_demo(args, frame_provider, args.height_size, args.cpu, args.track, args.smooth)

    pass

def create_cloth_mask(paths):
    print(paths["cloth"], paths["cloth_mask"])
    grabcut( input_dir = paths["cloth"], output_dir = paths["cloth_mask"] )

def create_val_list_segmentation(paths):
    image_name = paths["image_ind"].split("/")[-1].split(".")[0]
    nxb_img   = Image.open(paths["image_ind"])
    ref_img  = Image.open(paths["sample_edge"])
    nxb_ref_img  = ref_img.resize(nxb_img.size, Image.NEAREST)
    nxb_ref_img.save( os.path.join( paths["seg_edge"], image_name + ".png" ) )

    ref_img  = Image.open(paths["sample_label"])
    nxb_ref_img  = ref_img.resize(nxb_img.size, Image.NEAREST)
    nxb_ref_img.save( os.path.join( paths["seg_label"], image_name + ".png" ) )
    
    text = paths["image_ind"] + " " +  os.path.join( paths["seg_label"], image_name + ".png" )
    text2 =  image_name

    with open( os.path.join(paths["seg_list"] ,"val.txt") ,"w") as f:
        f.write(text)
    with open( os.path.join(paths["seg_list"] ,"val_id.txt") ,"w") as f:
        f.write(text2)
    

def create_segmentation(paths):
    segment_images(DATA_DIR = paths["base"], LIST_PATH = os.path.join(paths["base"], "list", "val.txt"), DATA_ID_LIST = os.path.join(paths["base"], "list", "val_id.txt"), RESTORE_FROM = "./semanticParsingCheckpoint/CIHP_pgn", parsing_dir = paths["image_parse"])
    pass

def create_image_mask(paths):
    image_name = paths["image_ind"].split("/")[-1].split(".")[0]
    im_parse = Image.open(os.path.join(paths["image_parse"], image_name + ".png")) # read segmentation
    parse_array = np.array(im_parse) # convert to numpy array
    parse_shape = (parse_array > 0).astype(np.float32) # get binary body shape
    cv2.imwrite(os.path.join(paths["image_mask"] , image_name + ".png"), parse_shape)
    print(image_name)

def create_swap_list(paths):
    path = os.path.dirname(paths["base"])
    text = paths["image_ind"].split("/")[-1] + " " + paths["cloth_ind"].split("/")[-1]
    with open( os.path.join(path ,"test_pairs.txt") ,"w") as f:
        f.write(text)


def swap_cloths(paths):
    create_swap_list(paths)
    opt = get_opt(paths, "GMM", "GMM", "checkpoints/GMM/gmm_final.pth")
    print(opt)
    output(opt)
    opt = get_opt(paths, "TOM", "TOM", "checkpoints/TOM/tom_final.pth")
    print(opt)
    output(opt)
def main(person_image_path, cloth_image_path, sample_edge = "./sample_edge.png", sample_label = "./sample_label.png"):
    paths = {}
    paths["base"] = os.path.dirname(os.path.dirname(person_image_path) )
    paths["image_ind"] = person_image_path
    paths["cloth_ind"] = cloth_image_path
    paths["image"] = create_directory(os.path.join(paths["base"], "image"))
    paths["cloth"] = create_directory(os.path.join(paths["base"], "cloth"))
    

    paths["image_mask"] = create_directory(os.path.join(paths["base"] , "image-mask"))

    paths["sample_edge"] = sample_edge
    paths["sample_label"] = sample_label

    paths["seg_edge"] = create_directory(os.path.join(paths["base"] , "edges"))
    paths["seg_label"] = create_directory(os.path.join(paths["base"] , "labels"))
    paths["seg_list"]= create_directory(os.path.join(paths["base"] , "list"))
    paths["image_parse"] = create_directory(os.path.join(paths["base"] , "image-parse-new"))

    paths["key_pose"] = create_directory(os.path.join(paths["base"], "pose"))

    paths["cloth_mask"] = create_directory(os.path.join(paths["base"], "cloth-mask"))

    
    create_cloth_mask(paths)
    create_keypoints(paths)
    create_val_list_segmentation(paths)
    create_segmentation(paths)
    create_image_mask(paths)
    swap_cloths(paths)


if __name__ == "__main__":
    main("./users/user_id/test/image/000003_0.jpg", "./users/user_id/test/cloth/000004_1.jpg")


    


