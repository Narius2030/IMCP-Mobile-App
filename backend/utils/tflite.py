import tensorflow as tf

def convert_to_tflite():
    # Tải mô hình Keras đã huấn luyện
    keras_model = tf.keras.models.load_model("./utils/yolo8-bert-lstm/model_yolo_bert_lstm_14ep_cp-0001.h5")

    # Chuyển đổi mô hình sang định dạng TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()

    # Lưu mô hình TFLite vào file
    with open('./utils/yolo8-bert-lstm/model_yolo_bert_lstm_14ep_cp-0001.tflite', 'wb') as f:
        f.write(tflite_model)
        
        
def check_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="./utils/vgg16-lstm/img_caption_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("input details", input_details)
    print("ouput details", output_details)
    
    
if __name__ == '__main__':
    convert_to_tflite()