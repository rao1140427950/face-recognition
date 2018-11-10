import cv2 as cv
import tensorflow as tf
import numpy as np

CV_PATH='D:/Anaconda/envs/tensorflow_cpu/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'  # your cv file path
MODE_SAVE_PATH='Face Recognition/mode_save'
IMAGE_PATH='Face Recognition/train_set/0.jpg'
TEMP_PATH='Face Recognition/train_set/temp/temp.jpg'

# Define placeholder to feed training data
_x=tf.placeholder(tf.float32,[None,200,200,3])
x_image=tf.reshape(_x,[-1,200,200,3])

# Conv layer 1
filter1=tf.Variable(tf.truncated_normal([11,11,3,96],stddev=0.0001))
bias1=tf.Variable(tf.truncated_normal([96],stddev=0.0001))
conv1=tf.nn.conv2d(x_image,filter1,strides=[1,4,4,1],padding='SAME')
conv1=tf.nn.relu(tf.add(conv1,bias1))

# Pool layer 1
pool1=tf.nn.avg_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# LRN layer 1: Local response normalization
norm1=tf.nn.lrn(pool1,5,bias=1,alpha=0.001/9.0,beta=0.75)

# Conv layer 2
filter2=tf.Variable(tf.truncated_normal([5,5,96,256],stddev=0.01))
bias2=tf.Variable(tf.truncated_normal([256],stddev=0.1))
conv2=tf.nn.conv2d(norm1,filter2,strides=[1,1,1,1],padding='SAME')
conv2=tf.nn.relu(tf.add(conv2,bias2))

# Pool layer 2
pool2=tf.nn.avg_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
# LRN layer 2:
norm2=tf.nn.lrn(pool2,5,bias=1,alpha=0.001/9.0,beta=0.75)

# Conv layer 3
filter3=tf.Variable(tf.truncated_normal([3,3,256,384],stddev=0.01))
bias3=tf.Variable(tf.truncated_normal([384],stddev=0.1))
conv3=tf.nn.conv2d(norm2,filter3,strides=[1,1,1,1],padding='SAME')
conv3=tf.nn.relu(tf.add(conv3,bias3))

# Conv layer 4
filter4=tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.01))
bias4=tf.Variable(tf.truncated_normal([384],stddev=0.1))
conv4=tf.nn.conv2d(conv3,filter4,strides=[1,1,1,1],padding='SAME')
conv4=tf.nn.relu(tf.add(conv4,bias4))

# Conv layer 5
filter5=tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.01))
bias5=tf.Variable(tf.truncated_normal([256],stddev=0.1))
conv5=tf.nn.conv2d(conv4,filter5,strides=[1,1,1,1],padding='SAME')
conv5=tf.nn.relu(tf.add(conv5,bias5))

# Pool layer 5
pool5=tf.nn.avg_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
pool5=tf.reshape(pool5,[-1,6*6*256])

# Link layer 1
w1=tf.Variable(tf.truncated_normal([6*6*256,4096],stddev=0.1))
b1=tf.Variable(tf.truncated_normal([4096],stddev=0.1))
y1=tf.add(tf.matmul(pool5,w1),b1)
y1=tf.nn.relu(y1)
#y1=tf.nn.dropout(y1,0.5)

# Link layer 2
w2=tf.Variable(tf.truncated_normal([4096,2048],stddev=0.1))
b2=tf.Variable(tf.truncated_normal([2048],stddev=0.1))
y2=tf.add(tf.matmul(y1,w2),b2)
y2=tf.nn.relu(y2)
#y2=tf.nn.dropout(y2,0.5)

# Link layer3, classify layer
w3=tf.Variable(tf.truncated_normal([2048,2],stddev=0.1))
b3=tf.Variable(tf.truncated_normal([2],stddev=1))
y_pred=tf.nn.softmax(tf.add(tf.matmul(y2,w3),b3))

#OpenCV face recognition
face_cascade=cv.CascadeClassifier(CV_PATH)
font=cv.FONT_HERSHEY_SIMPLEX

#Open camera and csv file
cam=cv.VideoCapture(0)

#Init trained model
saver=tf.train.Saver()
with tf.Session() as sess:
	
	#Get saved model
	save_model=tf.train.latest_checkpoint(MODE_SAVE_PATH)
	#Restore saved model
	saver.restore(sess,save_model)

	for i in range(0,10000):
		ret,frame=cam.read()  #Take a phote
		faces=face_cascade.detectMultiScale(frame)  #Detect faces

		for (x,y,w,h) in faces:

			temp=max(w,h)
			img=frame[y:(y+temp),x:(x+temp),:];  #Get faces
			# imgg=img[:,:,0]
			# img[:,:,0]=img[:,:,2]
			# img[:,:,2]=imgg

			#To recognize
			img=cv.resize(img,(200,200),interpolation=cv.INTER_LINEAR)
			image=tf.cast(img,tf.float32)
			image=tf.image.per_image_standardization(image)
			image=tf.reshape(image,[1,200,200,3])
			image=sess.run(image)
			pred=sess.run(y_pred,feed_dict={_x:image})

			pred=np.reshape(pred,[2])
			max_index=np.argmax(pred)
			
			if pred[max_index]>0.9:
				if max_index==0:  # If it's Rao
					cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
					frame=cv.putText(frame,'Rao',(x,y),font,1.5,(255,0,0),2)
				if max_index==1:
					cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
					frame=cv.putText(frame,'Tian',(x,y),font,1.5,(0,255,0),2)

			cv.imshow('Face',frame)
			cv.waitKey(10)

cam.release()


