

import os
import json
import argparse
import network_helper as nh


parser = argparse.ArgumentParser(description='Predict a flower name from an image along with '
                                             'the probability of that name.',
                                 usage='python predict.py ./data_sets/valid/1/img.jpg')
parser.add_argument('image_path', type=str, help='Path to image to predict')
parser.add_argument('checkpoint', type=str, help='Checkpoint file location')
parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training, defaults to False')
parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                    help='Mapping file of class id to real names, defaults to cat_to_name.json',)
args = parser.parse_args()


def main(args):
    use_gpu: bool = args.gpu
    print('Using GPU') if use_gpu else print('Using CPU')

    # Load category->class mappings
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model, optimizer = nh.load_checkpoint(checkpoint_path=args.checkpoint)
    acc, classes = nh.predict(image_path=args.image_path, model=model,
                              categories=cat_to_name, topk=args.top_k, use_gpu=use_gpu)
    print('---------------Results-----------------')
    for i in range(args.top_k):
        print(f'Prediction: {cat_to_name[classes[i]].upper()}({classes[i]}) with accuracy {acc[i]:.5f}')
    print('---------------------------------------')


if __name__ == '__main__':
    main(args)
