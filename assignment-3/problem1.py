from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open(fp = "1080p.jpg")
plt.imshow(img)
plt.title("Image")
plt.axis("off")
plt.show()
print(f"Format: {img.format}, Pixels: {img.size}, Mode: {img.mode}")

#original array
img_arr = np.array(img)
new_arr = np.roll(img_arr, shift=-1, axis=1)
img2 = Image.fromarray(new_arr)
plt.imshow(img2)
plt.title("Image")
plt.axis("off")
plt.show()

img_arr = img_arr.reshape(-1, 1)
new_arr = new_arr.reshape(-1, 1)

print(f"L1 Loss: {np.sum(np.abs(img_arr-new_arr))}")
print(f"L2 Loss: {np.sum((img_arr-new_arr)**2)}")

# print(f"L1 Loss: {np.sum(np.abs(img_arr-img_arr))}")
# print(f"L2 Loss: {np.sum((img_arr-img_arr)**2)}")