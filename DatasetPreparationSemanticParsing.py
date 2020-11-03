import numpy as np
import cv2
import glob
import os

from PIL import Image

class semantic_dataloader():
    def __init__(self, data_dir, edge_dir, label_dir, list_dir,):
        self.image_list = glob.glob(data_dir + "/images/*.*g")
        self.data_dir = data_dir
        self.edge_dir = edge_dir
        self.label_dir = label_dir
        self.list_dir = list_dir
        self.create_edges()
        self.create_labels()
        

    def create_edges(self):
        for image_path in self.image_list:
            image_name = image_path.split("/")[-1].split(".")[0]
            if not os.path.exists( os.path.join( self.edge_dir, image_name + ".png" ) ):
                print(image_name)
                nxb_img   = Image.open(image_path)
                ref_img  = Image.open(os.path.join( self.data_dir + "/sample_edge.png" ))
                nxb_ref_img  = ref_img.resize(nxb_img.size, Image.NEAREST)
                
                
                assert( nxb_img.size == nxb_ref_img.size)



    def create_labels(self):
        for image_path in self.image_list:
            image_name = image_path.split("/")[-1].split(".")[0]
            if not os.path.exists( os.path.join( self.label_dir, image_name + ".png" ) ):
                nxb_img   = Image.open(image_path)
                ref_img  = Image.open(os.path.join( self.data_dir + "/sample_label.png" ))
                nxb_ref_img  = ref_img.resize(nxb_img.size, Image.NEAREST)
                nxb_ref_img.save( os.path.join( self.label_dir, image_name + ".png" ) )
                assert( nxb_img.size == nxb_ref_img.size)
    
    def create_text_file(self, file_name, file_name_id,):

        

        with open( os.path.join(self.list_dir, file_name ) , "w") as f:
            with open(os.path.join(self.list_dir , file_name_id) , "w") as f2:
                for image_path, label_path in zip( sorted(glob.glob(self.data_dir + "/images/*.*g")), sorted(glob.glob(self.label_dir + "/*.*g") )):

                    # print(image_path.replace("\\", "/") + " " + label_path.replace("\\", "/") + "\n")
                    f.write(image_path.replace("\\", "/") + " " + label_path.replace("\\", "/") + "\n")
                    image_name = image_path.split("/")[-1].split(".")[0]
                    f2.write(image_name + "\n")

if __name__ == "__main__":
    d = semantic_dataloader( "./datasets_segmentation/CIHP", "./datasets_segmentation/CIHP/edges", "./datasets_segmentation/CIHP/labels" , "./datasets_segmentation/CIHP/list")
    d.create_text_file("val.txt", "val_id.txt")
