from __future__ import print_function

import logging
import os
import json

from dgmvae import evaluators, utt_utils
from dgmvae import main as main_train
from dgmvae import main_aggresive as main_train_agg
from dgmvae.dataset import corpora
from dgmvae.dataset import data_loaders
from dgmvae.dataset import biodataloader
from dgmvae.models.sent_models import *
from dgmvae.utils import prepare_dirs_loggers, get_time
from dgmvae.multi_bleu import multi_bleu_perl
from dgmvae.options import get_parser

logger = logging.getLogger()
""" 
def get_corpus_client(config):
    if config.data.lower() == "ptb":
        corpus_client = corpora.PTBCorpus(config)
    elif config.data.lower() == "daily_dialog":
        corpus_client = corpora.DailyDialogCorpus(config)
    elif config.data.lower() == "stanford":
        corpus_client = corpora.StanfordCorpus(config)
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")
    return corpus_client
 """
def get_corpus_client(config):
    if config.data.lower() == "neural":
        corpus_client = corpora.NeuroCorpus(config)
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")
    return corpus_client

def get_data_split(config):
    data_path = config.data_dir  # 指定数据文件所在目录
    # 从文件中读取数据
    data_file = os.path.join(data_path, 'sim_100d_poisson_disc_label.npz')
    data_input = np.load(data_file)  # 读取 npz 文件
    u_true = data_input['u']
    z_true = data_input['z']
    x_true = data_input['x']
    # 将10000长度的时间序列切分成50*200
    x_all = x_true.reshape(50, 200, -1).transpose(1, 0, 2)  # 将(50, 200, -1)转置为(200, 50, -1)
    u_all = u_true.reshape(50, 200, -1).transpose(1, 0, 2)
    z_all = z_true.reshape(50, 200, -1).transpose(1, 0, 2)



    x_train = x_all[:, :40, :] # 前40个时间点训练集
    u_train = u_all[:, :40, :]
    z_train = z_all[:, :40, :]


    x_valid = x_all[:, 40:45, :] # 5个做验证集
    u_valid = u_all[:, 40:45, :]
    z_valid = z_all[:, 40:45, :]

    x_test = x_all[:, 45:, :] # 5个做测试集
    u_test = u_all[:, 45:, :]
    z_test = z_all[:, 45:, :]
    
    # 将训练集、验证集和测试集保存为字典
    train_data = {'x': x_train, 'u': u_train, 'z': z_train}
    valid_data = {'x': x_valid, 'u': u_valid, 'z': z_valid}
    test_data = {'x': x_test, 'u': u_test, 'z': z_test}

    # 返回字典形式的数据集
    return train_data, valid_data, test_data



def get_dataloader(config, data_split):
    if config.data.lower() == "neural":
        dataloader = biodataloader.DataLoader

    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")

    train_dial, valid_dial, test_dial = data_split[0], \
                                        data_split[1], \
                                        data_split[2]

    train_feed = dataloader("Train", config, train_dial)
    valid_feed = dataloader("Valid", config, valid_dial)
    test_feed = dataloader("Test", config, test_dial)

    return train_feed, valid_feed, test_feed

def get_model(config): # corpus_client这个参量可以去掉，模型训练不需要
    try:
        model = eval(config.model)(config) # eval(config.model) 的作用是根据 config.model 中的字符串名称来获取对应的模型类，并创建一个实例。
    # 如果模型初始化失败就抛出异常
    except Exception as e:
        raise NotImplementedError("Fail to build model %s" % (config.model))
    if config.use_gpu:
        model.cuda()
    return model

def evaluation(model, test_feed, train_feed, evaluator):
    if config.aggressive:
        engine = main_train_agg
    else:
        engine = main_train
    # 如果 forward_only 为 True，则将文件保存在日志目录下。否则，将文件保存在会话目录下。    
    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
        sampling_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-sampling.txt".format(get_time()))
    else:
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.session_dir, "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")
        sampling_file = os.path.join(config.session_dir, "{}-sampling.txt".format(get_time()))

    # 将批量大小设置为 50，并加载之前保存的模型参数（model_file）。 
    config.batch_size = 50
    model.load_state_dict(torch.load(model_file))

    # 调用 engine.validate 函数来计算模型在测试集和验证集上的性能。
    engine.validate(model, test_feed, config)
    # engine.validate(model, valid_feed, config)

    # if hasattr(model, "sampling_for_likelihood"):
    #     nll = utt_utils.calculate_likelihood(model, test_feed, 500, config)  # must
    #     if config.forward_only:
    #         logger_file = open(os.path.join(config.log_dir, config.load_sess, "session.log"), "a")
    #         logger_file.write("Log-likehood %lf" % nll)

    # 调用 utt_utils.find_mi 函数来计算并打印模型的同质性得分。这个函数通常用于衡量聚类结果的质量，可能会计算如均匀性（homogeneity）等指标。
    """
    print("--test homogeneity--")
    utt_utils.find_mi(model, test_feed, config)  # homogeneity_score

    # with open(os.path.join(sampling_file), "w") as f:
    #     print("Saving test to {}".format(sampling_file))
    #     utt_utils.exact_sampling(model, 46000, config, dest_f=f)

    # 调用 utt_utils.find_mi 函数来计算并打印模型的同质性得分。这个函数通常用于衡量聚类结果的质量，可能会计算如均匀性（homogeneity）等指标。
    selected_clusters = utt_utils.latent_cluster(model, train_feed, config, num_batch=None)
    # 保存聚类结果和潜在空间数据
    with open(os.path.join(dump_file + '.json'), 'w') as f:
        json.dump(selected_clusters, f, indent=2)

    with open(os.path.join(dump_file), "wb") as f:
        print("Dumping test to {}".format(dump_file))
        utt_utils.dump_latent(model, test_feed, config, f, num_batch=None)

    with open(os.path.join(test_file), "w") as f:
        print("Saving test to {}".format(test_file))
        utt_utils.generate(model, test_feed, config, evaluator, num_batch=None, dest_f=f)
    # 计算模型生成文本的 BLEU 分数，这是一种常用的文本生成质量评估指标。通过 multi_bleu_perl 函数来评估模型在生成任务上的表现。
    multi_bleu_perl(config.session_dir if not config.forward_only else os.path.join(config.log_dir, config.load_sess))
    """
def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))

    data_split = get_data_split(config)
    evaluator = evaluators.BleuEvaluator("CornellMovie") # 这个后续需要修改
    train_feed, valid_feed, test_feed = get_dataloader(config, data_split)
    #train_batch = train_feed.epoch_init(config, shuffle=True)
    #train_batch = train_feed.next_batch()
    # 输出批次数据
    #print(train_batch['x'].shape)

    model = get_model(config)

    if config.forward_only is False:
        try:
            if config.aggressive:
                engine = main_train_agg
            else:
                engine = main_train
            # 训练
            engine.train(model, train_feed, valid_feed,
                         test_feed, config, evaluator, gen=utt_utils.generate)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")
    # 验证， 根据情况修改
    evaluation(model, test_feed, train_feed, evaluator)

if __name__ == "__main__":
    config = get_parser()
    with torch.cuda.device(config.gpu_idx):
        main(config)
