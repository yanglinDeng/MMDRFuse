
class args():

    # training args
    epochs = 10
    batch_size = 25
    trainNumber = 16000
    HEIGHT = 128
    WIDTH = 128
    in_c = 2
    out_c = 1

    Patch_path = "train_imgs/LLVIP_patches/ " # path to training image patches
    save_stumodel_dir = "models/student" # path to save student model
    save_stuloss_dir = "models/student/loss" # path to save loss of student model

    save_teamodel_dir = "models/teacher"  # path to save teacher model
    save_tealoss_dir = "models/teacher/loss"  # path to save loss of teacher model

    cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
    seed = 42 #"random seed for training"

    lr = 1e-4 #"learning rate, default is 1e-4"
    log_interval = 2 #"number of images after which the training loss is logged, default is 500"
    device = 0;

    pretrained_teacher_path = "pretrained_models/teacher.model"
    pretrained_student_path = "pretrained_models/student.model"

    #settings for test
    test_imgs_path = ""
    save_outputs_path = ""
    test_nums = ""





