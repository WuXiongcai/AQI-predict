import argparse
from train import main as train_main
from test import main as test_main

def main():
    parser = argparse.ArgumentParser(description='空气质量预测系统')
    parser.add_argument('mode', choices=['train', 'test'],
                      help='选择运行模式：train（训练模型）或 test（预测未来24小时）')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("开始训练模型...")
        train_main()
    else:
        print("开始预测未来24小时的空气质量...")
        test_main()

if __name__ == '__main__':
    main()
