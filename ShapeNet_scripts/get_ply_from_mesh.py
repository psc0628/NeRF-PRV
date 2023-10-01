import pymeshlab
import os
import numpy
import threading
import time

thread_num_max = 1
g_num = 0
# 创建互斥锁,默认不上锁
mutex = threading.Lock()

def sampling_thread_process(num,models_path,model_path,model_file,ply_file,pcd_file,vox_file):
    print(model_file + ' working, the num is ' + str(num))
    # use meshlab to get ply
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_file)
    ms.apply_filter('compute_color_from_texture_per_vertex')
    ms.save_current_mesh(ply_file, binary=False, save_vertex_normal=False, save_wedge_texcoord=False,save_face_color=False)
    # wait mesh_sample to get ply points
    ms.load_new_mesh(vox_file)
    ms.save_current_mesh(pcd_file, binary=False, save_vertex_normal=False, save_wedge_texcoord=False,save_face_color=False)
    #os.remove(vox_file)

    global g_num
    mutex.acquire()  # 上锁
    g_num -= 1
    mutex.release()  # 解锁

#'04379243','02958343','03001627','02691156','04256520','04090263','03636649','04530566','02828884','03691459','02933112','03211117','04401088'
objects = ['04379243','02958343','03001627','02691156','04256520','04090263','03636649','04530566','02828884','03691459','02933112','03211117','04401088']

if __name__ == '__main__':
    num = 0
    for object in objects:
        num_catlog = 0
        models_path = 'D:/VS_project/ShapeNetCore.v2/' + object + '/'
        for model_path in os.listdir(models_path):
            # consider the model has texture image
            if os.path.exists(models_path + model_path + '/images/') == True:
                num_catlog = num_catlog + 1
                if num_catlog > 1200:
                    break

                model_file = models_path + model_path + '/models/model_normalized.obj'
                ply_file = models_path + model_path + '/models/model_normalized.ply'
                pcd_file = models_path + model_path + '/models/model_normalized_sample.ply'
                vox_file = models_path + model_path + '/models/vox_sampled_mesh_0.ply'
                if os.path.exists(pcd_file) == True:
                    print(pcd_file + ' exist and skip')
                    continue
                if os.path.exists(vox_file) == False:
                    print(vox_file + ' not exist. Run get_mesh_sampling first.')
                    continue

                if os.path.exists(model_file) == False:
                    print(model_file + ' not exist and skip this folder')
                    continue

                num = num + 1

                # 多线程
                mutex.acquire()
                g = g_num
                mutex.release()
                while g == thread_num_max:
                    time.sleep(1)
                    mutex.acquire()
                    g = g_num
                    mutex.release()
                mutex.acquire()
                g_num += 1
                mutex.release()
                threading.Thread(target=sampling_thread_process,args=(num, models_path, model_path,model_file,ply_file,pcd_file,vox_file)).start()

                # 单线程
                #ms = pymeshlab.MeshSet()
                #ms.load_new_mesh(model_file)
                #ms.apply_filter('compute_color_from_texture_per_vertex')
                #ms.save_current_mesh(ply_file, binary=False, save_vertex_normal=False, save_wedge_texcoord=False, save_face_color=False)
                #ms.load_new_mesh(vox_file)
                #ms.save_current_mesh(pcd_file, binary=False, save_vertex_normal=False, save_wedge_texcoord=False, save_face_color=False)
                #os.remove(vox_file)

    