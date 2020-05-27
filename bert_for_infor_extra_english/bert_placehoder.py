# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/4/23
import tensorflow as tf
import modeling,argparse,random,os
import tokenization
import numpy as np
from sklearn import model_selection
from tensorflow.contrib.rnn import LSTMCell
import pandas as pd
from TJl_function import get_logger


#获得输入bert模型的数据
def data_process(text,selected_text,sentiment,tokenize,max_len):
    def get_train_label(text,selected_text):
        char=np.zeros(len(text))
        idx=text.find(selected_text)
        char[idx:idx+len(selected_text)]=1
        assert sum(char)>0
        return char
    text=' '.join(text.split()).lower()
    selected_text=' '.join(selected_text.split()).lower()
    char = get_train_label(text, selected_text)

    split_words=tokenize.tokenize(text)
    length=len(split_words)
    idx=0
    offsets=[]
    for i in range(length):
        #判断第i个单词前面是否为空格,若为空格idx+1跳过空格。offset只记录text中非空格的word的索引
        if text[idx]==' ':
            if split_words[i][:2]=='##':
                offset = (idx + 1, idx + 1 + len(split_words[i])-2)
                idx = idx + 1 + len(split_words[i])-2
            else:
                offset=(idx+1,idx+1+len(split_words[i]))
                idx=idx+1+len(split_words[i])
        else:
            if split_words[i][:2]=='##':
                offset = (idx, idx + len(split_words[i])-2)
                idx = idx + len(split_words[i])-2
            else:
                offset=(idx,idx+len(split_words[i]))
                idx=idx+len(split_words[i])
        offsets.append(offset)
    #当有[unk]时，则不相等
    try:
        assert offsets[-1][1]==len(text)
        assert len(offsets)==len(split_words)
    except:
        pass
    start_label=np.zeros(len(split_words),dtype=int)
    end_label=np.zeros(len(split_words),dtype=int)
    index=[]
    for i, (a, b) in enumerate(offsets):
        if sum(char[a:b])>0:index.append(i)
    start_label[index[0]]=1
    end_label[index[-1]]=1
    split_words=['[CLS]']+[sentiment]+['[SEP]']+['[CLS]']+split_words+['[SEP]']
    input_id=tokenize.convert_tokens_to_ids(split_words)
    mask_id=[1]*len(input_id)
    type_id=[0]*3+[0]*(len(input_id)-3)
    start_label=[0]*4+list(start_label)+[0]
    end_label=[0]*4+list(end_label)+[0]
    offsets=[(0,0)]*4+offsets+[(0,0)]
    if len(input_id)<max_len:
        input_id.extend((max_len-len(input_id))*[0])
        mask_id.extend((max_len-len(mask_id))*[0])
        type_id.extend((max_len-len(type_id))*[0])
        start_label.extend((max_len-len(start_label))*[0])
        end_label.extend((max_len-len(end_label))*[0])
        offsets.extend((max_len-len(offsets))*[(0,0)])
    else:
        assert len(input_id)<max_len
    assert(len(input_id)==len(mask_id))
    return (input_id,mask_id,type_id,start_label,end_label,offsets,text,selected_text,sentiment)


def batch_yield(data, batch_size, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)
    input_ids, mask_ids, type_ids, start_labels, end_labels, offsets,texts,selected_texts,sentiments = [], [], [], [], [], [],[],[],[]
    for (text, selected_text, sentiment) in data:
        input_id, mask_id, type_id, start_label, end_label, offset,text,selected_text,sentiment = data_process(text, selected_text, sentiment,
                                                                                  tokenize, args.max_len)
        input_ids.append(input_id)
        mask_ids.append(mask_id)
        type_ids.append(type_id)
        start_labels.append(start_label)
        end_labels.append(end_label)
        offsets.append(offset)
        texts.append(text)
        selected_texts.append(selected_text)
        sentiments.append(sentiment)
        if len(input_ids) == batch_size:
            assert (len(input_ids) == len(offsets))
            yield input_ids, mask_ids, type_ids, start_labels, end_labels, offsets,texts,selected_texts,sentiments
            input_ids, mask_ids, type_ids, start_labels, end_labels, offsets, texts,selected_texts,sentiments= [], [], [], [], [], [],[],[],[]

    if len(input_ids) != 0:
        yield input_ids, mask_ids, type_ids, start_labels, end_labels, offsets,texts,selected_texts,sentiments


def decode_prediction(pred_start, pred_end, text, offset, sentiment):
    def decode(pred_start, pred_end, text, offset):

        decoded_text = ""
        for i in range(pred_start, pred_end + 1):
            decoded_text += text[offset[i][0]:offset[i][1]]
            #判断该单词后是否要接空格
            if (i + 1) < len(offset) and offset[i][1] < offset[i + 1][0]:
                decoded_text += " "
        return decoded_text.strip()

    decoded_predictions = []
    for i in range(len(text)):
        if sentiment[i] == "neutral" or len(text[i].split()) < 3:
            decoded_text = text[i]
        else:
            idx_start = np.argmax(pred_start[i])
            idx_end = np.argmax(pred_end[i])
            if idx_start > idx_end:
                idx_end = idx_start
            decoded_text = str(decode(idx_start, idx_end, text[i], offset[i]))
            if len(decoded_text) == 0:
                decoded_text = text[i]
        decoded_predictions.append(decoded_text)

    return decoded_predictions

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


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
class BertModel(object):
    def __init__(self,args):
        self.clip_grad=args.clip #gradient clipping
        self.max_len=args.max_len
        self.batch_size=args.batch_size
        self.model_path=args.model_save_path
    def build_graph(self):
        self.add_placeholder()
        self.add_bert_model()
        self.add_bilstm()
        self.add_dense()
        self.add_loss()
        self.add_optimizer()
        self.add_init()
    def add_placeholder(self):
        self.input_id=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.type_id=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.mask_id=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.start_label=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.end_label=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.keep_rate=tf.placeholder(dtype=tf.float32, shape=[], name="keep_rate")
    def add_bert_model(self):

        self.bert_config = modeling.BertConfig.from_json_file(args.bert_config_path)
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=True,
            input_ids=self.input_id,
            input_mask=self.mask_id,
            token_type_ids=self.type_id,
            use_one_hot_embeddings=False)
        # load_weights(init_checkpoint)
        self.bert_embeddings = model.get_sequence_output()
        load_weights(args.bert_path)
    def add_bilstm(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(256)
            cell_bw = LSTMCell(256)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.bert_embeddings,
                #sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            self.bilstm_embeddings = tf.nn.dropout(output, self.keep_rate)
    def add_dense(self):
        shape=tf.shape(self.bilstm_embeddings)
        W = tf.get_variable(name="W",
                            shape=[512, 2],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)

        b = tf.get_variable(name="b",
                            shape=[2],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        self.bilstm_embeddings=tf.reshape(self.bilstm_embeddings,[-1,shape[-1]])
        self.bilstm_embeddings=tf.nn.dropout(self.bilstm_embeddings, self.keep_rate)
        pred=tf.matmul(self.bilstm_embeddings,W)+b
        pred=tf.reshape(pred,[-1,args.max_len,2])
        self.start_logits, self.end_logits=tf.split(pred,2,axis=-1)
        self.start_logits = tf.squeeze(self.start_logits, axis=-1)
        self.end_logits = tf.squeeze(self.end_logits, axis=-1)
    def add_loss(self):
        self.loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.start_logits,labels=self.start_label)
        self.loss+=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.end_logits,labels=self.end_label)
        self.loss = tf.reduce_mean(self.loss)
    def add_optimizer(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars if g is not None]
        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
    def add_init(self):
        self.init_op = tf.global_variables_initializer()

    def train(self,train_data,valid_data,test_data,fold_num):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:

            sess.run(self.init_op)
            # variables = tf.contrib.framework.get_variables_to_restore()
            # saver.restore(sess, args.bert_path)
            # num_batches = (len(data) + self.batch_size - 1) // self.batch_size
            best_score=-1
            for epoch in range(args.epochs):
                generator_batch=batch_yield(train_data, args.batch_size, shuffle=True)
                loss_epoch=0
                for steps,(input_ids, mask_ids, type_ids, start_labels, end_labels, offsets,_,_,_)in enumerate(generator_batch):
                    # if steps==0:args.learn_rate=1e-7
                    # if steps==1:args.learn_rate=3e-5
                    feed_dict={self.input_id:input_ids,
                               self.mask_id:mask_ids,
                               self.type_id:type_ids,
                               self.start_label:start_labels,
                               self.end_label:end_labels,
                               self.lr_pl:args.learn_rate,
                               self.keep_rate: args.keep_rate
                               }
                    _, loss_train, step_num_,e = sess.run([self.train_op, self.loss, self.global_step,self.bert_embeddings],
                                                                 feed_dict=feed_dict)
                    # print(input_ids[0])
                    # print(e)
                    # print(step_num_)
                    # exit()
                    loss_epoch+=loss_train
                    if steps%100==0:
                        loginfo('epoch: {} loss_train:{:.4}'.format(epoch+1,loss_epoch/(steps+1)))
                _,_,_,_,_,_,score,loss=self.evaluate(sess=sess,data=train_data)
                loginfo('train: epoch: {} jac_score: {} loss: {}'.format(epoch+1,score,loss))
                loginfo('#'*100)
                _,_,_,_,_,_,score, loss = self.evaluate(sess=sess, data=valid_data)
                loginfo('valid: epoch: {} jac_score: {} loss: {}'.format(epoch+1,score,loss))
                if not os.path.exists(self.model_path+'%d/'%fold_num):os.makedirs(self.model_path+'%d/'%fold_num)
                if score>best_score:
                    best_score=score
                    saver.save(sess, self.model_path+'%d/'%fold_num+'ckpt', global_step=self.global_step)
                    if len(test_data)>0:
                        pred_start,pred_end,text,selected_text,sentiment,offset,_,_ = self.evaluate(sess=sess, data=test_data)
        if len(test_data)>0:
            return pred_start,pred_end,text,selected_text,sentiment,offset
    def evaluate(self,sess,data):
        generator_batch = batch_yield(data, args.batch_size, shuffle=False)
        # Initialize accumulators
        offset = []
        text = []
        selected_text = []
        sentiment = []
        pred_start = []
        pred_end = []
        losses=[]
        for steps, (input_ids, mask_ids, type_ids, start_labels, end_labels, offsets,texts,selected_texts,sentiments) in enumerate(generator_batch):
            feed_dict = {self.input_id: input_ids,
                         self.mask_id: mask_ids,
                         self.type_id: type_ids,
                         self.start_label: start_labels,
                         self.end_label: end_labels,
                         self.keep_rate:1
                         }
            start_logits,end_logits, loss_eval = sess.run([self.start_logits,self.end_logits, self.loss],
                                                feed_dict=feed_dict)
            # add batch to accumulators
            pred_start.extend(start_logits)
            pred_end.extend(end_logits)
            offset.extend(offsets)
            text.extend(texts)
            selected_text.extend(selected_texts)
            sentiment.extend(sentiments)
            losses.append(loss_eval)
        selected_text_pred = decode_prediction(
            pred_start, pred_end, text, offset, sentiment)
        # print(selected_text_pred)
        jaccards = []
        for i in range(len(selected_text)):
            jaccards.append(
                jaccard(selected_text[i], selected_text_pred[i]))
        score = np.mean(jaccards)
        loss_avg=sum(losses)/len(losses)
        return pred_start,pred_end,text,selected_text,sentiment,offset,score,loss_avg
    def predict(self,test_data,model_path=None):
        if not model_path:
            raise ValueError('Invalid model_path')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            generator_batch = batch_yield(test_data, args.batch_size, shuffle=False)
            # Initialize accumulators
            offset = []
            text = []
            selected_text = []
            sentiment = []
            pred_start = []
            pred_end = []
            for steps, (input_ids, mask_ids, type_ids, start_labels, end_labels, offsets, texts, selected_texts,
                        sentiments) in enumerate(generator_batch):
                feed_dict = {self.input_id: input_ids,
                             self.mask_id: mask_ids,
                             self.type_id: type_ids,
                             self.start_label: start_labels,
                             self.end_label: end_labels
                             }
                start_logits, end_logits = sess.run([self.start_logits, self.end_logits],
                                                               feed_dict=feed_dict)
                # add batch to accumulators
                pred_start.extend(start_logits)
                pred_end.extend(end_logits)
                offset.extend(offsets)
                text.extend(texts)
                selected_text.extend(selected_texts)
                sentiment.extend(sentiments)
            selected_text_pred = decode_prediction(
                pred_start, pred_end, text, offset, sentiment)
        return selected_text_pred
if __name__ == '__main__':
    # data = [("it is  sooo nice i like very much! ,！ ！", ' sooo nice', 'positive'),
    #         ("it is so bad...,i can't like it!!!", 'so bad...', 'negative'),
    #         ("it is so bad nice. . .,i can't like it", 'so bad', 'negative'),
    #         ("it is so nice bad,i can like it", 'so nice', 'positive'),
    #         ("coool!!!,it can't be better","coool!!!,it can't be better",'neutral'),
    #         ("I feel [] useless** I don't know what to do right now. I'm so bored",'useless','negative')
    #         ]
    # test_data=data
    parser = argparse.ArgumentParser(description='bert for english text extract')
    parser.add_argument('--max_len', default=128, help='句子的最大长度')
    parser.add_argument('--batch_size', default=16, help='训练批次大小')
    parser.add_argument('--epochs', default=5, help='训练整批数量的回合')
    parser.add_argument('--keep_rate', default=0.8, help='keep_rate')
    parser.add_argument('--learn_rate', default=1e-5, help='bert模型训练初始学习率')
    parser.add_argument('--num_folds', type=int, default=5, help='N折')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--bert_path', type=str, default='uncased_L-12_H-768_A-12/bert_model.ckpt', help='预训练bert模型的路径')
    parser.add_argument('--vocab_path', type=str, default='uncased_L-12_H-768_A-12/vocab.txt', help='vocab的路径')
    parser.add_argument('--bert_config_path', type=str, default='uncased_L-12_H-768_A-12/bert_config.json',
                        help='bert_config的路径')
    parser.add_argument('--init_checkpoint', type=str, default='model_save_path/', help='bert_config的路径')
    parser.add_argument('--model_save_path', type=str, default='output/model/', help='模型保存的路径')
    parser.add_argument('--logs_path', type=str, default='output/', help='模型保存的路径')
    args = parser.parse_args()
    # 构建分词器
    tokenize = tokenization.FullTokenizer(args.vocab_path, do_lower_case=True)
    if not os.path.exists(args.logs_path):os.makedirs(args.logs_path)
    loginfo=get_logger(args.logs_path+'log.txt')
    train_data=pd.read_csv('data/train.csv')
    train_data=train_data.dropna()
    test_data=pd.read_csv('data/test.csv')
    test_data['selected_text']=test_data.text
    submission=pd.read_csv('data/sample_submission.csv')
    data=list(zip(train_data.text,train_data.selected_text,train_data.sentiment))

    test_data=list(zip(test_data.text,test_data.selected_text,test_data.sentiment))

    model=BertModel(args=args)
    model.build_graph()
    kfold = model_selection.KFold(
        n_splits=args.num_folds, shuffle=True, random_state=42)
    data=np.array(data)
    test_data=np.array(test_data)
    start_preds=np.zeros((args.max_len,len(test_data)))
    end_preds=np.zeros((args.max_len,len(test_data)))
    for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(data)):
        # if fold_num>0:
        #     break
        loginfo("\nfold %02d" % (fold_num + 1))
        train_data=data[train_idx]
        valid_data=data[valid_idx]
        pred_start,pred_end,text,selected_text,sentiment,offset=model.train(
            train_data=train_data,valid_data=valid_data,test_data=test_data,fold_num=fold_num)
        start_preds+=pred_start
        end_preds+=pred_end
    selected_text_pred = decode_prediction(
        start_preds, end_preds, text, offset, sentiment)
    print(selected_text_pred)
    print(len(start_preds))
    submission.selected_text=selected_text_pred
    submission.to_csv('submission.csv',index=None,header=True)


    # pred=model.predict(test_data=data)