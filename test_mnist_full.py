import os


def test_main():
    print('##############全连接手写体网络##############')
    script_path = 'SIT/mnist_full_dpk/script/'

    print('--------------训练 开始--------------')
    train_ret = os.popen(f'python {script_path}/component/snn/snn_train.py').read()
    if "train finish." in train_ret:
        print('--------------训练 结束--------------')
    else:
        raise RuntimeError(train_ret)

    print('--------------仿真器运行 开始--------------')
    main_ret = os.popen(f'python {script_path}/main_emu.py').read()
    if "correct" in main_ret:
        print('--------------仿真器运行 结束--------------')
    else:
        raise RuntimeError(main_ret)
