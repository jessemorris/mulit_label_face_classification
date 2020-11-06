from fastai.vision.all import *
from fastai.vision.augment import aug_transforms
import fastai
import pandas as pd
import imutils
import glob
import cv2
import shutil
from tqdm.notebook import tqdm
print(fastai.__version__)
pd.set_option('display.max_columns', 500)
tqdm().pandas()

import numpy as np
import os

package_path = os.path.abspath(os.getcwd())

dataset_path = package_path + "/dataset/"
processed_path = package_path + "/processed_data/"

print("package_path " + package_path)
print("dataset_path " + dataset_path)
print("processed_path " + processed_path)

def validation_func(x):
    return 'validation' in x

def pre_processing():

    ## Finding all the images in the folder
    # dataset/img_align_celeba/img_align_celeba
    item_list = glob.glob(dataset_path + '/img_align_celeba/img_align_celeba/*.jpg')

    #sort item list
    item_list = sorted(item_list, key=lambda x: x.split("/")[-1].split(".")[0])
    print(item_list[:20])
    

    generic_image_path = dataset_path + '/img_align_celeba/img_align_celeba/'

    item_evaluation = pd.read_csv(dataset_path + 'list_eval_partition.csv')
    for index, row in item_evaluation.iterrows():
        image_id = row["image_id"]
        image_path = generic_image_path + image_id

        #train-0, val-1, test-2
        partition = row["partition"]

        if int(partition) == 0:
            destination = processed_path+'training/'
        elif int(partition) == 1:
            destination = processed_path+'validation/'
        else:
            destination = processed_path+'testing/'

        #will do move later
        shutil.copy(
                image_path, 
                destination+image_id
            )

        print("index {}".format(index))
        print("row {} {}".format(row["image_id"],row["partition"]))

        if index > 50:
            break
    print(len(item_list))
    print(item_list[0])


    ## Findign all the images and separating in training and validation
    # item_list = glob.glob(path+'*.jpg')

    # for idx in tqdm(range(1,10000)):
    #     if idx <= 182637:
    #         destination = processed_path+'training/'
    #     else:
    #         destination = processed_path+'validation/'
    #     try:
    #         shutil.move(
    #             path+str(idx).zfill(6)+'.jpg', 
    #             destination+str(idx).zfill(6)+'.jpg'
    #         )
    #     except:
    #         print("shutil exception")
    #         pass
    #     # print("Processing {}/{}".format(idx,"10000"))


    label_df = pd.read_csv(dataset_path + 'list_attr_celeba.csv')
    print("Reading labels from: " + dataset_path + 'list_attr_celeba.csv')
    column_list = pd.Series(list(label_df.columns)[1:10])
    # print(column_list)

    def label_generator(row):
        return(' '.join(column_list[[True if int(i)==1 else False for i in row[column_list]]]))

    label_df['labels'] = label_df.apply(lambda x: label_generator(x), axis=1)
    label_df = label_df.loc[:,['image_id','labels']]
    print("label DF")
    print(label_df)
    # label_df = label_df.loc[:,['image_id','label']]
    #just writing the labels
    print("Writing labels to " + processed_path +  'labels.csv')
    # label_df.to_csv(processed_path +  'labels.csv')

    ## Attachhing label to correct file names
    # item_list = glob.glob(dataset_path + '/img_align_celeba/img_align_celeba/*.jpg')
    item_df = pd.DataFrame({'image_name':pd.Series(item_list[1:50]).apply(lambda x: x.split('/')[-1])})
    item_df['image_id'] = item_df.image_name
    print(item_df) 


    ## Creating final label set
    # label_df = pd.read_csv(processed_path +  'labels.csv')
    label_df = label_df.merge(item_df, on='image_id', how='inner')
    label_df.rename(columns={'image_name':'fname'}, inplace=True)
    label_df.loc[:,['fname','labels']].to_csv(processed_path +  'labels.csv', index=False)




if __name__ == "__main__":
    # pre_processing()
    # tfms = aug_transforms(pad_mode='zeros', mult=2, min_scale=0.5)
    # # tfms =  get_transforms(do_flip=False, flip_vert=False, max_rotate=30, max_lighting=0.3)
    df = pd.read_csv(processed_path +'labels.csv')
    print(df.head())

    dls = ImageDataLoaders.from_df(df, processed_path, folder='training/',label_delim=" ", header=df.head())

    dls.show_batch()
    #batch_tfms=aug_transforms(size=224)

    # # src = (ImageItemList.from_csv(path, csv_name=dataset_path +'list_attr_celeba.csv')
    # #    .split_by_valid_func(validation_func)
    # #    .label_from_df(cols='tags',label_delim=' '))

    # data = (src.transform(tfms, size=128)
    #     .databunch(bs=256).normalize(imagenet_stats))    