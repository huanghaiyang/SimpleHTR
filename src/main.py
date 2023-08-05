import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt' # 字符集
    fn_summary = '../model/summary.json' # 命中及错误率
    fn_corpus = '../data/corpus.txt' # 语料库

# 获取图片高度 NN模型固定高度为32
# 神经网络（NN）模型高度取决于多个因素，包括输入数据的质量和数量、网络架构的复杂程度、激活函数的选择、训练数据的多样性和数量、优化算法的类型和参数等。
# 一般情况下，更复杂的神经网络模型通常具有更高的预测准确性，但也更容易出现过拟合现象。因此，需要在设计神经网络模型时平衡模型的复杂度和可泛化性。
def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

# 获取图片尺寸
def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    # NN模型固定高度为32，宽度根据训练模式（单词或文本行）设置 128*32 或 256*32
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

# 写入训练结果 summary.json 文件
# 训练结果 summary.json 文件包含了训练过程中的平均损失、字符错误率和单词准确率。
# 该文件可以用于可视化训练过程中的损失和准确率。
# 该文件还可以用于比较不同模型的性能。
# 该文件的内容如下所示：
# {
#   "averageTrainLoss": [0.0, 0.0, 0.0, 0.0, 0.0],
#   "charErrorRates": [0.0, 0.0, 0.0, 0.0, 0.0],
#   "wordAccuracies": [0.0, 0.0, 0.0, 0.0, 0.0]
# }
# 如果训练过程中出现错误，可以通过删除该文件来重新开始训练。
# 还可以通过删除该文件来重新开始训练，但是这样会丢失之前的训练结果。
# 为了避免这种情况，可以将该文件的内容保存到另一个文件中，然后在训练过程中将其加载到内存中。
def write_summary(average_train_loss: List[float], char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'averageTrainLoss': average_train_loss, 'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)

# 读取字符集
def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

# 训练模型
def train(model: Model,
          loader: DataLoaderIAM, # IAM数据集加载器
          line_mode: bool, # 训练模式（单词或文本行）
          early_stopping: int = 25) -> None: # 及早停止
    """Trains NN."""
    # 训练重试次数
    epoch = 0  # number of training epochs since start
    # 错误率
    summary_char_error_rates = []
    # 准确率
    summary_word_accuracies = []  

    # 训练过程中的平均损失
    train_loss_in_epoch = []
    # 平均损失
    average_train_loss = []

    # 预处理器
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    # 最佳字符错误率
    best_char_error_rate = float('inf')  # best validation character error rate
    # 无改进次数
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    
    # 如果再过 x 个 epoch 没有改进，则停止训练
    while True:
        epoch += 1 # 训练重试次数加1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        # 随机选择训练子集
        loader.train_set()
        # 从数据集中获取下一个批次
        while loader.has_next():
            # 获取迭代器信息  
            iter_info = loader.get_iterator_info()
            # 获取下一个批次
            batch = loader.get_next()
            # 批次预处理
            batch = preprocessor.process_batch(batch)
            # 训练完成后返回损失率
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
            # 保存当前批次的损失率
            train_loss_in_epoch.append(loss)

        # validate
        # 验证 NN 模型 的准确率 和 错误率
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        # 将训练过程中的平均损失、字符错误率和单词准确率写入 summary.json 文件
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        # 计算平均损失
        average_train_loss.append((sum(train_loss_in_epoch)) / len(train_loss_in_epoch))
        # 将训练过程中的平均损失、字符错误率和单词准确率写入 summary.json 文件
        write_summary(average_train_loss, summary_char_error_rates, summary_word_accuracies)

        # reset train loss list
        # 重置训练损失列表
        train_loss_in_epoch = []

        # if best validation accuracy so far, save model parameters
        # 如果目前为止的最佳验证准确率，保存模型参数
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            # 重置当前的最佳字符错误率
            best_char_error_rate = char_error_rate
            # 重置无改进次数
            no_improvement_since = 0
            # 保存模型参数
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {best_char_error_rate * 100.0}%')
            # 无改进次数加1
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        # 如果最近 x 个 epoch 没有更多改进，则停止训练
        if no_improvement_since >= early_stopping:
            print(f'No more improvement for {early_stopping} epochs. Training stopped.')
            break


# 验证 NN 模型 的准确率 和 错误率
def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    # 验证集
    loader.validation_set()
    # 预处理器 处理图片 
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    # 字符错误率
    num_char_err = 0
    # 字符总数
    num_char_total = 0
    # 单词正确数
    num_word_ok = 0
    # 单词总数
    num_word_total = 0
    # 从数据集中获取下一个批次
    while loader.has_next():
        # 获取迭代器信息
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        # 获取下一个批次
        batch = loader.get_next()
        # 批次预处理
        batch = preprocessor.process_batch(batch)
        # 预测批次
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        # 识别结果
        for i in range(len(recognized)):
            # 如果识别结果和真实结果相同，则正确单词数加1
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            # 单词总数加1
            num_word_total += 1
            # 计算识别单词和真实单词的字符差异个数
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            # 字符错误数
            num_char_err += dist
            # 字符总数
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    # 计算字符错误率
    char_error_rate = num_char_err / num_char_total
    # 计算单词准确率
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy

# 识别图片中的文字
def infer(model: Model, fn_img: Path) -> None:
    """Recognizes text in image provided by file path."""
    # 读取图片 先转换为灰度图 再进行预处理 
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    # 判断图片是否为空
    assert img is not None

    # 预处理图片 动态调整图片宽度 使用16像素填充
    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    # 预处理图片
    img = preprocessor.process_img(img)

    # 批次
    batch = Batch([img], None, 1)
    # 识别图片中的文字 返回识别结果和识别概率
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"') 
    print(f'Probability: {probability[0]}')

# 解析命令行参数
def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    # 选择训练、验证或识别模式 默认为识别模式
    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    # 选择解码器 默认为 bestpath 解码器 可选 beamsearch 或 wordbeamsearch 其中 beamsearch 和 wordbeamsearch 需要安装 ctcdecode 库 
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    # 批次大小 默认为100
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    # IAM 数据集目录
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    # 从 LMDB 加载样本 默认为 False 从文件系统加载样本 如果为 True 则从 LMDB 加载样本 
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    # 训练模式 默认为 False 单词模式 如果为 True 则为文本行模式
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    # 识别图片路径 默认为 ../data/word.png
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    # 及早停止 默认为 25次识别错误后停止训练
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    # 是否将 NN 输出转储到 CSV 文件中 默认为 False
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


# 主函数 实现训练、验证和识别模式
def main():
    """Main function."""

    # parse arguments and set CTC decoder
    # 解析命令行参数 并设置 CTC 解码器
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath, # bestpath 解码器表示使用最佳路径解码器 按照最大概率选择最佳路径 最大概率路径是指最大化所有时间步的概率
                       'beamsearch': DecoderType.BeamSearch, # beamsearch 解码器表示使用束搜索解码器 束搜索解码器是一种贪婪解码器 它在每个时间步选择最有可能的字符
                       'wordbeamsearch': DecoderType.WordBeamSearch} # wordbeamsearch 解码器表示使用单词束搜索解码器 如果识别结果包含多个单词 则使用字典来限制识别结果 只有在字典中的单词才会被接受
    # 解码器类型
    decoder_type = decoder_mapping[args.decoder]

    # train the model
    # 如果是训练模式
    if args.mode == 'train':
        #  从指定目录加载 IAM 数据集
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

        # when in line mode, take care to have a whitespace in the char list
        # 如果是文本行模式 则在字符集中添加空格
        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters and words
        # 将字符集和语料库写入文件
        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        # 将训练集和验证集中的单词写入语料库
        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        # 训练模型
        model = Model(char_list, decoder_type)
        # 开始训练
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    # 如果是验证模式
    elif args.mode == 'validate':
        # 从指定目录加载 IAM 数据集
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        # 验证 NN 模型 的准确率 和 错误率
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        # 验证 NN 模型 的准确率 和 错误率
        validate(model, loader, args.line_mode)

    # infer text on test image
    # 如果是识别模式
    elif args.mode == 'infer':
        # 识别图片中的文字
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        # 识别图片中的文字
        infer(model, args.img_file)


if __name__ == '__main__':
    main()
