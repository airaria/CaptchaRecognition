from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import chain
import pickle

TARGET_HEIGHT=48
TARGET_WIDTH=128
MAXLEN = 6

number = [str(i) for i in range(10)]
alphabet = [chr(i) for i in range(ord('a'),ord('z')+1)]
Alphabet = [chr(i) for i in range(ord('A'),ord('Z')+1)]
alphabet.remove('o')
Alphabet.remove('O')
charset = number+alphabet+Alphabet


def random_chars(charset, nb_chars):
    return [np.random.choice(charset) for i in range(nb_chars)]

def gen_captcha(charset,nb_chars=None,font=None):
    if not font is None:
        image = ImageCaptcha(fonts=[font])

    buffer_index=1000
    buffer_size=1000
    nc_set = np.zeros(buffer_size)


    while True:
        if buffer_index==buffer_size:
            nc_set = np.random.randint(3, MAXLEN+1, buffer_size) if nb_chars is None else np.array([nb_chars] * buffer_size)
            buffer_index=0
        captcha_text = ''.join(random_chars(charset,nc_set[buffer_index]))
        buffer_index+=1

        img_text = ' '*np.random.randint(0,MAXLEN+1-len(captcha_text))*2+captcha_text #用空格模拟偏移
        captcha = image.generate(img_text)
        captcha_image = Image.open(captcha).resize((TARGET_WIDTH,TARGET_HEIGHT),Image.ANTIALIAS)
        #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
        captcha_array = np.array(captcha_image)
        yield captcha_array,captcha_text

def convert_to_npz(num,captcha_generator,is_encoded,is_with_tags):

    vocab = charset[:]
    if is_encoded:
        vocab += [' ']
    if is_with_tags:
        id2token = {k+1:v for k,v in enumerate(vocab)}
        id2token[0] = '^'
        id2token[len(vocab)+1]='$'
    else:
        id2token = dict(enumerate(vocab))

    token2id = {v:k for k,v in id2token.items()}

    vocab_dict ={"id2token":id2token,"token2id":token2id}
    with open("data/captcha.vocab_dict","wb") as dict_file:
        pickle.dump(vocab_dict,dict_file)
    fn = "data/captcha.npz"

    print("Writing ",fn)
    img_buffer = np.zeros((num,TARGET_HEIGHT,TARGET_WIDTH,3),dtype=np.uint8)
    text_buffer = []
    for i in range(num):
        x,y = next(captcha_generator)
        img_buffer[i] = x
        if is_with_tags:
            y = ("^"+y+"$")
        if is_encoded:
            text_buffer.append([token2id[i] for i in y.ljust(MAXLEN+2*is_with_tags)])
        else:
            text_buffer.append(y)
    np.savez(fn,img=img_buffer,text=text_buffer)
    return vocab_dict,img_buffer,text_buffer

def convert_to_tfrecord(num,captcha_generator,is_encoded,is_with_tags):
    vocab = charset
    if is_encoded:
        vocab += [" "]
    if is_with_tags:
        id2token = {k+1:v for k,v in enumerate(vocab)}
        id2token[0] = '^'
        id2token[len(vocab)]='$'
    else:
        id2token=dict(enumerate(vocab))

    token2id = {v:k for k,v in id2token.items()}
    vocab_dict ={"id2token":id2token,"token2id":token2id}
    with open("data/captcha.vocab_dict","wb") as dict_file:
        pickle.dump(vocab_dict,dict_file)
    fn = "data/captcha.tfrecords"
    print('Writing ',fn)
    writer = tf.python_io.TFRecordWriter(fn)
    for i in range(num):
        x,y = next(captcha_generator)
        if is_with_tags:
            y = "^" + y + "$"
        if is_encoded:
            y=np.array([token2id[i] for i in y.ljust(MAXLEN+2*is_with_tags)],dtype=np.int32)

            h,w = x.shape[:2]
            xb = x.tobytes()
            yb =y.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "height":tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                "width":tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[xb])),
                'text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[yb]))
            })
            )
        else:
            h,w = x.shape[:2]
            xb = x.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "height":tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                "width":tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[xb])),
                'text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.encode('utf8')]))
            })
            )
        writer.write(example.SerializeToString())
    writer.close()
    return vocab_dict

def read_tfreacod_nograph(fn,is_encoded):
    record_iterator = tf.python_io.tf_record_iterator(path=fn)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        img_string = (example.features.feature['img_raw'].bytes_list.value[0])
        text_string = (example.features.feature['text'].bytes_list.value[0])

        img = np.fromstring(img_string,dtype=np.uint8).reshape(height,width,3)
        if not is_encoded:
            text = text_string.decode('utf8')
        else:
            text = np.fromstring(text_string,dtype=np.int32)
        yield img,text

def read_tfrecord(fn,num_epochs,is_encoded):
    fn_queue = tf.train.string_input_producer([fn],num_epochs=num_epochs,shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fn_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
          'height': tf.FixedLenFeature([],tf.int64),
          'width': tf.FixedLenFeature([],tf.int64),
          'img_raw': tf.FixedLenFeature([], tf.string),
          'text': tf.FixedLenFeature([], tf.string),
      })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([height, width, 3])
    text = features['text']
    image = tf.reshape(tf.decode_raw(features['img_raw'], tf.uint8),image_shape)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=TARGET_HEIGHT,
                                           target_width=TARGET_WIDTH)
    if is_encoded:
        text = tf.reshape(tf.decode_raw(text,tf.int32),(MAXLEN,))
        print (text.shape)
    print (text.shape)
    return image, text

if __name__ == '__main__':
    captcha_generator = gen_captcha(charset,font='fonts/YaHeiConsolas.ttf')
    #x,y = next(captcha_generator)
    #plt.imshow(x)
    #plt.show()
    #print(y)
    vocab_dict,img,text = convert_to_npz(num=65536,captcha_generator=captcha_generator,
                   is_encoded=True,is_with_tags=True)
    #vocab_dict = convert_to_tfrecord(65536,captcha_generator,is_encoded=False,is_with_tags=True)

    '''
    img,text = read_tfrecord("./data/captcha.tfrecords",num_epochs=2,is_encoded=False)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    it,tt  =tf.train.shuffle_batch([img, text],batch_size=32,
                                    capacity=1024,num_threads=2,
                                    min_after_dequeue=128)
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                i,t = sess.run([it,tt])
                print (i.shape,t[0])
                #print(''.join([vocab_dict['id2token'][i] for i in t[0]]))
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
    '''
