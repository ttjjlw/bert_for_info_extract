# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date: 2020/5/3
import tensorflow as tf
import numpy as np
import os,re,argparse
import modeling,tokenization
from util import convert_to_tfrecord,file_based_input_fn_builder,decode_prediction
import pandas as pd
from sklearn import model_selection

def load_weights(init_checkpoint):
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    print("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        print("  name = %s, shape = %s%s", var.name, var.shape,
              init_string)
def my_model(features,labels,mode,params):
    args=params['args']
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    start_label = features["start_label"]
    end_label = features["end_label"]
    print('shape of input_ids', input_ids.shape)
    print('shape of start_label', start_label.shape)
    print('shape of end_label', end_label.shape)
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_path)
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )
    load_weights(args.bert_path)
    bert_embedding=model.get_sequence_output()
    dropout = tf.layers.dropout(
        inputs=bert_embedding, rate=args.drop_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense = tf.layers.dense(
        inputs=dropout, units=2, activation=tf.nn.relu)
    start_logits,end_logits=tf.split(dense,2,-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "start_pred": tf.argmax(input=start_logits, axis=1),
        "end_pred": tf.argmax(input=end_logits, axis=1),

    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimaotor.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=start_logits, labels=features["start_label"])
    loss += tf.losses.sparse_softmax_cross_entropy(logits=end_logits, labels=features["end_label"])
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops={'eval_loss':
                              (tf.metrics.mean_squared_error(labels=start_label, predictions=predictions['start_pred'])[0]
        + tf.metrics.mean_squared_error(labels=end_label, predictions=predictions['end_pred'])[0],tf.metrics.mean_squared_error(labels=end_label, predictions=predictions['end_pred'])[1])
                          }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optim = tf.train.AdamOptimizer(learning_rate=args.learn_rate)
        train_op = optim.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bert for english text extract')
    parser.add_argument('--max_len', default=128, help='句子的最大长度')
    parser.add_argument('--batch_size', default=16, help='训练批次大小')
    parser.add_argument('--epochs', default=5, help='训练整批数量的回合')
    parser.add_argument('--drop_rate', default=0.2, help='keep_rate')
    parser.add_argument('--learn_rate', default=5e-5, help='bert模型训练初始学习率')
    parser.add_argument('--save_checkpoints_steps', default=500, help='每隔多少步保存模型')
    parser.add_argument('--max_steps_without_increase', default=500, help='')
    parser.add_argument('--num_folds', type=int, default=5, help='N折')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--bert_path', type=str, default='uncased_L-12_H-768_A-12/bert_model.ckpt', help='预训练bert模型的路径')
    parser.add_argument('--vocab_path', type=str, default='uncased_L-12_H-768_A-12/vocab.txt', help='vocab的路径')
    parser.add_argument('--bert_config_path', type=str, default='uncased_L-12_H-768_A-12/bert_config.json',
                        help='bert_config的路径')
    parser.add_argument('--init_checkpoint', type=str, default='model_save_path/', help='bert_config的路径')
    parser.add_argument('--model_save_path', type=str, default='output/model/', help='模型保存的路径')
    parser.add_argument('--logs_path', type=str, default='output/', help='模型保存的路径')
    parser.add_argument('--output_dir', type=str, default='output/', help='模型保存的路径')
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    train_data = pd.read_csv('data/train.csv')
    train_data = train_data.dropna().reset_index(drop=True)
    print('train_data:{}'.format(train_data.shape))
    tokenize = tokenization.FullTokenizer(args.vocab_path, do_lower_case=True)
    test_data = pd.read_csv('data/test.csv')
    test_data['selected_text'] = test_data.text
    print('test_data:{}'.format(test_data.shape))
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    offset, text, sentiment=convert_to_tfrecord(test_data, tokenize, args.max_len, args.output_dir + 'test.tfrecord')
    test_input = file_based_input_fn_builder(args.output_dir + 'test.tfrecord', args.max_len, is_training=False,
                                              drop_remainder=False)
    submission = pd.read_csv('data/sample_submission.csv')


    kfold = model_selection.KFold(
        n_splits=args.num_folds, shuffle=True, random_state=42)
    start_preds = [np.zeros(args.max_len)] * len(test_data)
    end_preds = [np.zeros(args.max_len)] * len(test_data)
    for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(train_data)):
        train = train_data.iloc[train_idx]
        valid = train_data.iloc[valid_idx]
        num_train_steps = int(
            len(train_data) * 1.0 / args.batch_size * args.epochs)
        print('num_train_steps: %d'%num_train_steps)
        convert_to_tfrecord(train, tokenize, args.max_len, args.output_dir + 'train.tfrecord')
        convert_to_tfrecord(valid, tokenize, args.max_len, args.output_dir + 'valid.tfrecord')
        train_input=file_based_input_fn_builder(args.output_dir + 'train.tfrecord',args.max_len,is_training=True,drop_remainder=True)
        valid_input=file_based_input_fn_builder(args.output_dir + 'valid.tfrecord',args.max_len,is_training=False,drop_remainder=False)
        params = {
            'args': args
        }
        estimator=tf.estimator.Estimator(
            my_model,
            model_dir=args.output_dir,  # config和这里都可以设置模型保存路径，二选一设置即可,都设置必须保持一致
            params=params
        )
        early_stopping_hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator=estimator,
            metric_name='eval_loss',
            max_steps_without_increase=args.max_steps_without_increase,
            run_every_steps=args.save_checkpoints_steps,
            run_every_secs=None,#设置了run_every_steps必须把run_every_secs设为None
        )
        if not os.path.exists(os.path.join(args.output_dir, 'eval')):
            os.mkdir(os.path.join(args.output_dir, 'eval'))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=valid_input, steps=None,
                                          throttle_secs=120)  # steps 评估的迭代步数，如果为None，则在整个数据集上评估。每save一次model才会评估一次，并且至少间隔120秒
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        result = estimator.predict(input_fn=test_input)
        start_preds+=result['start_pred']
        end_preds+=result['end_pred']
    selected_text_pred = decode_prediction(
        start_preds, end_preds, text, offset, sentiment)
    print(selected_text_pred)
    print(len(start_preds))
    submission.selected_text = selected_text_pred
    submission.to_csv('submission.csv', index=None, header=True)