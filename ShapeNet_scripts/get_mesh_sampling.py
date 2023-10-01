import pymeshlab
import os
import numpy
import threading
import time

thread_num_max = 50
g_num = 0
# 创建互斥锁,默认不上锁
mutex = threading.Lock()

def sampling_thread_process(num,model_file,models_path,model_path):
    print(model_file + ' working, the num is ' + str(num))
    os.system('python mesh_sampling_geo_color_shapenet.py ' + model_file + ' ' + models_path + model_path + '/models/' + ' --cloudcompare_bin_path D:\Software\CloudCompare\CloudCompare.exe --target_points 500000 --resolution 1024')
    global g_num
    mutex.acquire()  # 上锁
    g_num -= 1
    mutex.release()  # 解锁


#'04379243','02958343','03001627','02691156','04256520','04090263','03636649','04530566','02828884','03691459','02933112','03211117','04401088','02924116','02808440','03467517','03325088','03046257','03991062','03593526'
objects = ['04379243','02958343','03001627','02691156','04256520','04090263','03636649','04530566','02828884','03691459','02933112','03211117','04401088','02924116','02808440','03467517','03325088','03046257','03991062','03593526']

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

                num = num + 1
                model_file = models_path + model_path + '/models/model_normalized.obj'
                vox_file = models_path + model_path + '/models/vox_sampled_mesh_0.ply'
                if os.path.exists(vox_file) == True:
                    print(vox_file + ' exist and skip')
                    continue

                # 多线程
                mutex.acquire()
                g = g_num
                mutex.release()
                while g == thread_num_max:
                    time.sleep(10)
                    mutex.acquire()
                    g = g_num
                    mutex.release()
                mutex.acquire()
                g_num += 1
                mutex.release()
                threading.Thread(target=sampling_thread_process, args=(num,model_file,models_path,model_path,)).start()

                # 单线程
                #print(model_file + ' working, the num is ' + str(num))
                #os.system('python mesh_sampling_geo_color_shapenet.py ' + model_file + ' ' + models_path + model_path + '/models/' + ' --cloudcompare_bin_path D:\Software\CloudCompare\CloudCompare.exe --target_points 500000 --resolution 1024')

    