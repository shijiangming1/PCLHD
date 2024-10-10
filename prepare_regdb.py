"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = 'data/regdb'

if not os.path.isdir(download_path):
    print('please change the download_path')

#-----------------------------------------

#-----------------------------------------
#query
# query_path = download_path + '/query'
mode1 ='/ir_modify/'
save_path_first = download_path + mode1
if not os.path.isdir(save_path_first):
    os.mkdir(save_path_first)
for trial in range(1,11):
    n=0
    save_path = download_path + mode1+str(trial)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    query_save_path = download_path + mode1+str(trial)+'/query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    #######################
    data_path=download_path

    test_file_path = os.path.join(data_path,'idx/test_thermal_'+str(trial)+'.txt')
    with open(test_file_path) as f:
        data_file_list = open(test_file_path, 'rt').read().splitlines()
        # Get full list of image and labels
        files_ir = [data_path + '/' + s for s in data_file_list]
        # print(files_ir)
                # new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                # files_ir.extend(new_files)
            # print(files_ir)
    exist_id = {}
    for file_path in files_ir:
        file_list = file_path.split('/')
        c_id = 'c1'
        ID  = file_path.split(' ')[1]
        # print(ID)
        img_name = file_list[-1].split(' ')[-0]
        # print(file_list)
        src_path = file_path.split(' ')[0]
        dst_path = query_save_path
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        name = ID+"_"+c_id+"_"+img_name
        exist_id[ID] = exist_id.get(ID,0)+1
        if exist_id[ID]<=4:
            # print(src_path,dst_path)
            copyfile(src_path, dst_path + '/' + name)
            print(dst_path + '/' + name)
mode1 ='/rgb_modify/'
save_path_first = download_path + mode1
if not os.path.isdir(save_path_first):
    os.mkdir(save_path_first)
for trial in range(1,11):
    n=0
    save_path = download_path + mode1+str(trial)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    query_save_path = download_path + mode1+str(trial)+'/query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    #######################
    data_path=download_path

    test_file_path = os.path.join(data_path,'idx/test_visible_'+str(trial)+'.txt')
    with open(test_file_path) as f:
        data_file_list = open(test_file_path, 'rt').read().splitlines()
        # Get full list of image and labels
        files_ir = [data_path + '/' + s for s in data_file_list]
        # print(files_ir)
                # new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                # files_ir.extend(new_files)
            # print(files_ir)
    exist_id = {}
    for file_path in files_ir:
        file_list = file_path.split('/')
        c_id = 'c1'
        ID  = file_path.split(' ')[1]
        img_name = file_list[-1].split(' ')[-0]
        src_path = file_path.split(' ')[0]
        dst_path = query_save_path
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        name = ID+"_"+c_id+"_"+img_name
        exist_id[ID] = exist_id.get(ID,0)+1
        if exist_id[ID]<=4:
            # print(name)
            copyfile(src_path, dst_path + '/' + name)
            print(dst_path + '/' + name)
            
mode1 ='/ir_modify/'
if not os.path.isdir(save_path_first):
    os.mkdir(save_path_first)
for trial in range(1,11):
    save_path = download_path + mode1+str(trial)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    query_save_path = download_path + mode1+str(trial)+'/bounding_box_test'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    #######################
    data_path=download_path

    test_file_path = os.path.join(data_path,'idx/test_thermal_'+str(trial)+'.txt')
    with open(test_file_path) as f:
        data_file_list = open(test_file_path, 'rt').read().splitlines()
        files_ir = [data_path + '/' + s for s in data_file_list]

    for file_path in files_ir:
        file_list = file_path.split('/')
        c_id = 'c1'
        ID  = file_path.split(' ')[1]

        img_name = file_list[-1].split(' ')[-0]

        src_path = file_path.split(' ')[0]
        dst_path = query_save_path
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        name = ID+"_"+c_id+"_"+img_name
        print(name)

        copyfile(src_path, dst_path + '/' + name)
        print(dst_path + '/' + name)
    # #################################
mode1 ='/rgb_modify/'
save_path_first = download_path + mode1
if not os.path.isdir(save_path_first):
    os.mkdir(save_path_first)
for trial in range(1,11):
    save_path = download_path + mode1+str(trial)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    query_save_path = download_path + mode1+str(trial)+'/bounding_box_test'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    #######################
    data_path=download_path

    test_file_path = os.path.join(data_path,'idx/test_visible_'+str(trial)+'.txt')
    with open(test_file_path) as f:
        data_file_list = open(test_file_path, 'rt').read().splitlines()

        files_ir = [data_path + '/' + s for s in data_file_list]


    for file_path in files_ir:
        file_list = file_path.split('/')
        c_id = 'c1'
        ID  = file_path.split(' ')[1]

        img_name = file_list[-1].split(' ')[-0]

        src_path = file_path.split(' ')[0]
        dst_path = query_save_path
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        name = ID+"_"+c_id+"_"+img_name
        # print(src_path,dst_path)
        copyfile(src_path, dst_path + '/' + name)
        print(dst_path + '/' + name)


    # #################################
mode1 ='/ir_modify/'
mode2 ='/rgb_modify/'
save_path_first = download_path + mode1
if not os.path.isdir(save_path_first):
    os.mkdir(save_path_first)
for trial in range(1,11):
    save_path = download_path + mode1+str(trial)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    query_save_path = download_path + mode1+str(trial)+'/bounding_box_train'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    #######################
    data_path=download_path

    test_file_path = os.path.join(data_path,'idx/train_thermal_'+str(trial)+'.txt')
    with open(test_file_path) as f:
        data_file_list = open(test_file_path, 'rt').read().splitlines()
        # Get full list of image and labels
        files_ir = [data_path + '/' + s for s in data_file_list]


    for file_path in files_ir:
        file_list = file_path.split('/')
        c_id = 'c1'
        ID  = file_path.split(' ')[1]
        img_name = file_list[-1].split(' ')[-0]
        src_path = file_path.split(' ')[0]
        dst_path = query_save_path
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        name = ID+"_"+c_id+"_"+img_name
        copyfile(src_path, dst_path + '/' + name)
        print(dst_path + '/' + name)
    save_path = download_path + mode2+str(trial)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    query_save_path = download_path + mode2+str(trial)+'/bounding_box_train'
    test_file_path = os.path.join(data_path,'idx/train_visible_'+str(trial)+'.txt')

    with open(test_file_path) as f:
        data_file_list = open(test_file_path, 'rt').read().splitlines()
        # Get full list of image and labels
        files_ir = [data_path + '/' + s for s in data_file_list]
    for file_path in files_ir:
        file_list = file_path.split('/')
        c_id = 'c1'
        ID  = file_path.split(' ')[1]
        img_name = file_list[-1].split(' ')[-0]
        src_path = file_path.split(' ')[0]
        dst_path = query_save_path
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        name = ID+"_"+c_id+"_"+img_name
        copyfile(src_path, dst_path + '/' + name)
        print(dst_path + '/' + name)
