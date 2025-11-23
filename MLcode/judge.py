import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 加载模型
model = tf.keras.models.load_model("C:\\Users\\es602\\Desktop\\code\\model_fold_5.keras")

# 类别名称 (根据你的训练数据)
class_names = ['Anthracnose', 'fruit_fly', 'healthy_guava' ]  # 替换为你的类别名称

# 图片路径

#img_path = "C:\\Users\\es602\\Desktop\\Guava\\test\\healthy_guava\\000094.png"  # 000001-00093
img_path = "C:\\Users\\es602\\Desktop\\Guava\\test\\Anthracnose\\000157.png" # 000001-000156
#img_path = "C:\\Users\\es602\\Desktop\\Guava\\test\\fruit_fly\\000042.png" # 000001-000132
# 载入和预处理图片
def preprocess_image(img_path, image_size=224):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    img_array = img_array / 255.0  # 归一化处理
    return img_array

# 预测函数
def predict_image(model, img_path, class_names):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# 运行预测
predicted_class, confidence = predict_image(model, img_path, class_names)

# 输出预测结果
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")

# 显示图片和结果
img = image.load_img(img_path)
plt.imshow(img)
plt.title(f"Predicted: {predicted_class} ({confidence}%)")
plt.axis("off")
plt.show()
