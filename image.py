from PIL import Image
from numpy import asarray
# load the image
image = Image.open('instrument_instances356656.png')
image3 = Image.open('instrument_instances96000.png')
# convert image to numpy array
data = asarray(image)
data2 = asarray(image3)
print(type(data))
print(type(data2))

# summarize shape
print(data.shape)
print(data2.shape)
image.show()
# # create Pillow image
# image2 = Image.fromarray(data)
# image4 = Image.fromarray(data2)
# print(type(image2))
# print(type(image4))
# # summarize image details
# print(image2.mode)
# print(image4.mode)


# print(image2.size)
# print(image4.size)
# print(data[0][0])

print(type(image))
if(0 in image[0][0]):
	print('true')


for x in range(image2.size[1]):
	for y in range(image2.size[0]):
		if(0 not in data[x][y]):
			if(255 not in data[x][y]):
				print(data[x][y])


