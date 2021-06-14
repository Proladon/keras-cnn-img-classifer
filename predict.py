from keras.models import load_model
import os
from keras_preprocessing import image
import numpy as np
# from rich import print

test_path = 'C:/Users/Proladon/Desktop/ml/predict' #要預測的目標圖片資料夾
model = load_model('models/maid_v_bunny_model_v2') #載入預先訓練的模型

classes = ['maid', 'bunny']

for i in os.listdir(test_path):
  img = image.load_img(test_path + '//' + i, target_size=(500, 500))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0) # 在第<axis>維，增加一個維度: (x, y) -> (x, y, z)
  images = np.vstack([x]) # 將圖片陣列垂直堆疊成圖片陣列集
  val = model.predict(images) #預測該圖片陣列集
  
  print(f"{i}\nmaid: {val[0][0]}\nbunny: {val[0][1]}\n")
  # print(f"[bold indian_red1]{i}[/bold indian_red1]")
  # print(f"[bold turquoise2]maid:[/bold turquoise2] [white]{val[0][0]}[/white]")
  # print(f"[bold spring_green1]bunny:[/bold spring_green1] [white]{val[0][1]}[/white]")
  # print("\n")