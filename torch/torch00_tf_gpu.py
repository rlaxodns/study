import tensorflow as tf
print("텐서플로 버젼", tf.__version__)

if tf.config.list_physical_devices('gpu'):
    print('gpu있음')

else:
    print('cpu')

# cuda 버젼
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print(cuda_version)

cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print(cudnn_version)