# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date: 2020/5/3
import numpy as np
import tensorflow as tf
import collections

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
    index=[]
    for i, (a, b) in enumerate(offsets):
        if sum(char[a:b])>0:index.append(i)
    start_label=index[0]
    end_label=index[-1]
    split_words=['[CLS]']+[sentiment]+['[SEP]']+['[CLS]']+split_words+['[SEP]']
    input_id=tokenize.convert_tokens_to_ids(split_words)
    mask_id=[1]*len(input_id)
    type_id=[0]*3+[0]*(len(input_id)-3)
    start_label+=4
    end_label+=4
    offsets=[(0,0)]*4+offsets+[(0,0)]
    if len(input_id)<max_len:
        input_id.extend((max_len-len(input_id))*[0])
        mask_id.extend((max_len-len(mask_id))*[0])
        type_id.extend((max_len-len(type_id))*[0])
        offsets.extend((max_len-len(offsets))*[(0,0)])
    else:
        assert len(input_id)<max_len
    assert(len(input_id)==len(mask_id))
    assert(type(start_label)==int and type(end_label)==int)
    return (input_id,mask_id,type_id,start_label,end_label,offsets,text,selected_text,sentiment)

# class InputFeatures(object):
#     """A single set of features of data."""
#
#     def __init__(self, input_ids, input_mask, segment_ids, start_label, end_label, offsets, text, selected_text, sentiment ):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.satrt_label = start_label
#         self.end_label=end_label
#         self.offset=offsets
#         self.text=text
#         self.selected_text=selected_text
#         self.sentiment=sentiment
def convert_to_tfrecord(raw_data,tokenize,max_len,tfrecord_file_path):
    writer = tf.python_io.TFRecordWriter(tfrecord_file_path)
    #遍历所有数据
    offset_lis,text_lis,sentiment_lis=[],[],[]#最后predict要用到
    for text,selected_text,sentiment in zip(raw_data.text,raw_data.selected_text,raw_data.sentiment):
        #针对一条数据进行处理
        input_id, mask_id, type_id, start_label, end_label, offsets, texts, selected_text, sentiments=data_process(text,selected_text,sentiment,tokenize,max_len)
        # feature = InputFeatures(input_id, mask_id, type_id, start_label, end_label, offsets, text, selected_text, sentiment)
        features = collections.OrderedDict()
        features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_id)))
        features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(mask_id)))
        features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(type_id)))
        features["start_label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[start_label]))
        features["end_label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[end_label]))
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        offset_lis.append(offsets)
        text_lis.append(texts)
        sentiment_lis.append(sentiments)
        return offset_lis,text,sentiment_lis
def file_based_input_fn_builder(tfrecord_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "start_label": tf.FixedLenFeature([], tf.int64),
        "end_label": tf.FixedLenFeature([], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            # if t.dtype == tf.int64:
            #     t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["args"].batch_size
        d = tf.data.TFRecordDataset(tfrecord_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
            #tf.contrib.data.map_and_batch
            #tf.data.experimental.map_and_batch
        d = d.apply(tf.contrib.data.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn

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