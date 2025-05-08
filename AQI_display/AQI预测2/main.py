import argparse
from train import main as train_main
from test import main as test_main

def main():
    parser = argparse.ArgumentParser(description='AQI预测系统')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                      help='运行模式: train (训练) 或 test (测试)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print('开始训练模型...')
        train_main()
    else:
        print('开始测试模型...')
        test_main()

if __name__ == '__main__':
    main()

