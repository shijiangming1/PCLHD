"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = 'data/sysu'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/ir_modify'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#query
# query_path = download_path + '/query'
query_save_path = download_path + '/ir_modify/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
#######################
data_path=download_path
ir_cameras = ['cam3','cam6']
test_file_path = os.path.join(data_path,'exp/test_id.txt')
files_rgb = []
files_ir = []
files_test=[]
with open(test_file_path, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    ids = ["%04d" % x for x in ids]
for id in sorted(ids):
    n=0
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            for single in os.listdir(img_dir):
                if n < 4:
                    files_ir.append(img_dir+'/'+single)
                    
                else:
                    files_test.append(img_dir+'/'+single)
                n=n+1
            # new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            # files_ir.extend(new_files)
        # print(files_ir)

for file_path in files_ir:
    file_list = file_path.split('/')
    ID  = file_list[-2]
    c_id = 'c'+file_list[-3][-1]
    img_name = file_list[-1]
    # print(file_list)
    src_path = file_path
    dst_path = query_save_path
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    name = ID+"_"+c_id+"_"+img_name
    copyfile(src_path, dst_path + '/' + name)

for file_path in files_test:
    file_list = file_path.split('/')
    ID  = file_list[-2]
    c_id = 'c'+file_list[-3][-1]
    img_name = file_list[-1]
    # print(file_list)
    src_path = file_path
    dst_path = download_path + '/ir_modify/bounding_box_test'
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    name = ID+"_"+c_id+"_"+img_name
    copyfile(src_path, dst_path + '/' + name)
    print(dst_path + '/' + name)
############################


query_save_path = download_path + '/ir_modify/bounding_box_train'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
#######################
data_path=download_path
ir_cameras = ['cam3','cam6']
# rgb_cameras = ['cam1','cam2','cam3','cam4','cam5','cam6']
test_file_path = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
files_rgb = []
files_ir = []
with open(test_file_path, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    ids_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
# print(id_val)
ids_train.extend(id_val) 
for id in sorted(ids_train):
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
        # print(files_ir)
############################
for file_path in files_ir:
    file_list = file_path.split('/')
    ID  = file_list[-2]
    c_id = 'c'+file_list[-3][-1]
    img_name = file_list[-1]
    # print(file_list)
    src_path = file_path
    dst_path = query_save_path
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    name = ID+"_"+c_id+"_"+img_name
    copyfile(src_path, dst_path + '/' + name)
    print(dst_path + '/' + name)
print(len(files_ir))

#-------------------------------------------------------

save_path = download_path + '/rgb_modify'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#query
# query_path = download_path + '/query'
query_save_path = download_path + '/rgb_modify/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
#######################
data_path=download_path
ir_cameras = ['cam1','cam2','cam4','cam5']
test_file_path = os.path.join(data_path,'exp/test_id.txt')
files_rgb = []
files_ir = []
files_test=[]
with open(test_file_path, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    ids = ["%04d" % x for x in ids]
for id in sorted(ids):
    n=0
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            for single in os.listdir(img_dir):
                if n < 4:
                    files_ir.append(img_dir+'/'+single)
                    
                else:
                    files_test.append(img_dir+'/'+single)
                n=n+1
            # new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            # files_ir.extend(new_files)
        # print(files_ir)

for file_path in files_ir:
    file_list = file_path.split('/')
    ID  = file_list[-2]
    c_id = 'c'+file_list[-3][-1]
    img_name = file_list[-1]
    # print(file_list)
    src_path = file_path
    dst_path = query_save_path
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    name = ID+"_"+c_id+"_"+img_name
    copyfile(src_path, dst_path + '/' + name)
    print(dst_path + '/' + name)
for file_path in files_test:
    file_list = file_path.split('/')
    ID  = file_list[-2]
    c_id = 'c'+file_list[-3][-1]
    img_name = file_list[-1]
    # print(file_list)
    src_path = file_path
    dst_path = download_path + '/rgb_modify/bounding_box_test'
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    name = ID+"_"+c_id+"_"+img_name
    copyfile(src_path, dst_path + '/' + name)
    print(dst_path + '/' + name)
############################


query_save_path = download_path + '/rgb_modify/bounding_box_train'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
#######################
data_path=download_path
ir_cameras = ['cam1','cam2','cam4','cam5']
# rgb_cameras = ['cam1','cam2','cam3','cam4','cam5','cam6']
test_file_path = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
files_rgb = []
files_ir = []
with open(test_file_path, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    ids_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
# print(id_val)
ids_train.extend(id_val) 
for id in sorted(ids_train):
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
        # print(files_ir)
print(len(files_ir))
############################
for file_path in files_ir:
    file_list = file_path.split('/')
    ID  = file_list[-2]
    c_id = 'c'+file_list[-3][-1]
    img_name = file_list[-1]
    # print(file_list)
    src_path = file_path
    dst_path = query_save_path
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    name = ID+"_"+c_id+"_"+img_name
    copyfile(src_path, dst_path + '/' + name)
    print(dst_path + '/' + name)
