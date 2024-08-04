# test_rgb phase
import torch
import numpy as np
from Evaluator import Evaluator,image_read_cv2
from PIL import Image
torch.set_default_tensor_type(torch.DoubleTensor)

def cal_metrics( output_path,infrared_path, visible_path, metrics):
    ir = image_read_cv2(infrared_path, 'GRAY')
    print("红外形状：{}".format(ir.shape))
    vi = image_read_cv2(visible_path, 'GRAY')
    print("可见光形状：{}".format(vi.shape))
    fi = image_read_cv2(output_path, 'GRAY')
    print("融合图像形状：{}".format(fi.shape))
    metrics += np.array([ Evaluator.SD(fi), Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi), Evaluator.Qabf(fi, ir, vi)
                            , Evaluator.SSIM(fi, ir, vi), Evaluator.CC(fi, ir, vi)])
    return metrics

def main():
    test_path = "original image path"
    output = "fused image path";
    num_runs = 1
    metric_result = np.zeros((6))
    with torch.no_grad():
        for i in range(0,num_runs):
            index = i + 1
            infrared_path = test_path + 'IR/' + str(index) + '.png'
            visible_path = test_path + 'VIS/' + str(index) + '.png'
            output_path = output + str(index) + '.png'
            metric_result += cal_metrics(output_path, infrared_path, visible_path, index)
    print('Done......')
    print("\t\t\t\t SD\t\tSCD\t\tVIF\t\tQabf\t\tSSIM\t\tCC")
    model_name = "MMDRFuse    "
    metric_result /= num_runs
    print(model_name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
          + str(np.round(metric_result[1], 2)) + '\t'
          + str(np.round(metric_result[2], 2)) + '\t'
          + str(np.round(metric_result[3], 2)) + '\t'
          + str(np.round(metric_result[4], 2)) + '\t'
          + str(np.round(metric_result[5], 2)) + '\t'
          )
    print("=" * 80)

if __name__ == '__main__':
    main()
