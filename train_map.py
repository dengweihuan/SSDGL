from simplecv import dp_train as train
import torch
from simplecv.util.logger import eval_progress, speed
import time
from module import SSDGL
from module import SSDGL_HOS
from simplecv.util import metric
from simplecv.util import registry
from torch.utils.data.dataloader import DataLoader
from simplecv import registry
from simplecv.core.config import AttrDict
from scipy.io import loadmat
import data.dataloader

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y



def fcn_evaluate_fn(self, test_dataloader, config):
    if self.checkpoint.global_step < 0:
        return
    self._model.eval()
    total_time = 0.
    y_all_list = []
    y_all_gt = []
    with torch.no_grad():
        for idx, (im, mask, w) in enumerate(test_dataloader):
            start = time.time()
            y_pred = self._model(im).squeeze()
            torch.cuda.synchronize()
            time_cost = round(time.time() - start, 3)
            y_pred = y_pred.argmax(dim=0).cpu() + 1
            w.unsqueeze_(dim=0)
            y_out = y_pred[0:610, 0:340]
            w = w.byte()

            mask = torch.masked_select(mask.view(-1), w.view(-1))

            y_pred = torch.masked_select(y_pred.view(-1), w.view(-1))

            gt_mat = loadmat('./pavia/PaviaU_gt.mat')
            gt_mask = gt_mat['paviaU_gt']

            gt = gt_mask.flatten()
            x_label = np.zeros(gt.shape)
            y_label = np.zeros(gt.shape)
            for i in range(len(gt)):
                if gt[i] == 0:
                    gt[i] = 17
                    x_label[i] = 16

            gt = gt[:] - 1
            y_out = y_out.flatten()
            for i in range(len(y_out)):
                if y_out[i] == 0:
                    y_out[i] = 17
                    y_label[i] = 16
            y_out = y_out[:] - 1
            x = np.ravel(y_out)
            y_list = list_to_colormap(x)
            y_gt = list_to_colormap(gt)
            y_all_list.append(y_list)
            y_all_gt.append(y_gt)
            y_re = np.reshape(y_list, (gt_mask.shape[0], gt_mask.shape[1], 3))
            gt_re = np.reshape(y_gt, (gt_mask.shape[0], gt_mask.shape[1], 3))

            classification_map(y_re, gt_mask, 300,
                               './classification_maps/' + str(0.05) + '_' + 'pavia.png')
            classification_map(gt_re, gt_mask, 300,
                               './classification_maps/' + str(0.05) + '_' + 'pavia_gt.png')

            oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
            aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 self._model.module.config.num_classes,
                                                                 return_accuracys=True)
            kappa = metric.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self._model.module.config.num_classes)
            total_time += time_cost
            speed(self._logger, time_cost, 'im')

            eval_progress(self._logger, idx + 1, len(test_dataloader))

    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batched im (avg)')

    metric_dict = {
        'OA': oa.item(),
        'AA': aa.item(),
        'Kappa': kappa.item()
    }
    for i, acc in enumerate(acc_per_class):
        metric_dict['acc_{}'.format(i + 1)] = acc.item()
    self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint.global_step)


def register_evaluate_fn(launcher):
    launcher.override_evaluate(fcn_evaluate_fn)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = train.parser.parse_args()
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    train.run(config_path=args.config_path,
              model_dir=args.model_dir,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[register_evaluate_fn],
              opts=args.opts)
