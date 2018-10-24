import os
import sys
from PIL import Image
import csv
import random
import numpy as np
import scipy.misc as smp
import time

def big_csv(stimuli_folder,save_out):
	emotion_dict={'AN':0,'DI':1,'AF':2,'HA':3,'SA':4,'SU':5,'NE':6}
	id_folders=list(os.walk(stimuli_folder))
	with open(save_out, 'w') as csvfile:
		stimwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		stimwriter.writerow(['Filepath','EmotionLabel','EmotionNumber','IdentityLabel','IdentityNumber'])
		for i in range(1,len(id_folders)):
			identity_folder=id_folders[i][0]
			id_label=os.path.basename(os.path.normpath(identity_folder))
			id_number=i-1
			for photo_file in os.listdir(identity_folder):
				emotion_label=photo_file[4:6]
				photo_file=os.path.join(identity_folder,photo_file)
				if photo_file.endswith("S.JPG"): 
					photo_name=os.path.basename(os.path.normpath(photo_file))
					stimwriter.writerow([photo_file,emotion_label,emotion_dict[emotion_label],id_label,id_number])

def leave_out_emotion(stimuli_folder,save_out,emotion_holdout):
	emotion_dict={'AN':0,'DI':1,'AF':2,'HA':3,'SA':4,'SU':5,'NE':6}
	id_folders=list(os.walk(stimuli_folder))
	with open(save_out, 'w') as csvfile:
		stimwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		stimwriter.writerow(['Filepath','IdentityNumber'])
		for i in range(1,len(id_folders)):
			identity_folder=id_folders[i][0]
			id_label=os.path.basename(os.path.normpath(identity_folder))
			id_number=i-1
			for photo_file in os.listdir(identity_folder):
				emotion_label=photo_file[4:6]
				photo_file=os.path.join(identity_folder,photo_file)
				if photo_file.endswith("S.JPG") and emotion_label !=emotion_holdout: 
					photo_name=os.path.basename(os.path.normpath(photo_file))
					stimwriter.writerow([photo_file,id_number])

def leave_out_identities(stimuli_folder,save_out,percent_holdout=0.1):
	emotion_dict={'AN':0,'DI':1,'AF':2,'HA':3,'SA':4,'SU':5,'NE':6}
	id_folders=list(os.walk(stimuli_folder))
	with open(save_out, 'w') as csvfile:
		stimwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		stimwriter.writerow(['Filepath','EmotionNumber'])
		i_vals=np.asarray(range(0,len(id_folders)-1))
		i_holdouts=np.ndarray.tolist(np.random.choice(i_vals,int(len(id_folders)*percent_holdout)))
		for i in range(1,len(id_folders)):
			if i not in i_holdouts:
				identity_folder=id_folders[i][0]
				id_label=os.path.basename(os.path.normpath(identity_folder))
				id_number=i-1
				for photo_file in os.listdir(identity_folder):
					emotion_label=photo_file[4:6]
					photo_file=os.path.join(identity_folder,photo_file)
					if photo_file.endswith("S.JPG"): 
						photo_name=os.path.basename(os.path.normpath(photo_file))
						stimwriter.writerow([photo_file,emotion_dict[emotion_label]])
def get_pix(photo_file):
	im = Image.open(photo_file)
	pix = im.load()
	width, height = im.size
	pixels=[0]*(width*height)
	#offset=round((height-width)/2)
	for x in range(width):
		for y in range(height):
			#pixels[y*width+x+2*offset*y]=(round(sum(pix[x,y])/3))
			pixels[y*width+x]=(round(sum(pix[x,y])/3))
	return pixels

def crop_image(image,points):
	top,bottom,left,right = points
	cropped_image=np.asarray(image[top:bottom,left:right])
	return(cropped_image.tolist())

def image_from_pix(pixels):
	image=np.asarray(pixels)
	I=np.resize(image,(762,562))
	I=crop_image(I,(272,272+388,87,87+388)) 
	#display_image(I,(388,388))
	return(I)

def display_image(pixels,size):
	image=np.asarray(pixels)
	I=np.resize(image,(size[0],size[1]))
	img = smp.toimage( I ) 
	img.show() 	

def lower_bound(stimuli_folder,save_out):
	emotion_dict={'AN':0,'DI':1,'AF':2,'HA':3,'SA':4,'SU':5,'NE':6}
	id_folders=list(os.walk(stimuli_folder))
	with open(save_out, 'w') as csvfile:
		stimwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		stimwriter.writerow(['emotion','pixels','Usage'])
		i_vals=np.asarray(range(0,len(id_folders)-1))
		#i_holdouts=np.ndarray.tolist(np.random.choice(i_vals,int(len(id_folders)*percent_holdout)))
		for i in range(1,len(id_folders)):
			#if i not in i_holdouts:
			identity_folder=id_folders[i][0]
			id_label=os.path.basename(os.path.normpath(identity_folder))
			id_number=i-1
			for photo_file in os.listdir(identity_folder):
				emotion_label=photo_file[4:6]
				photo_file=os.path.join(identity_folder,photo_file)
				if photo_file.endswith("S.JPG"):
					if id_label in ['AF01', 'AF02','AF03','BF01', 'BF02','BF03','AM01', 'AM02','AM03','BM01', 'BM02','BM03']:
						usage='ignore'
					if id_label in ['AF04', 'AF05','AF06','BF04', 'BF05','BF06','AM04', 'AM05','AM06','BM04', 'BM05','BM06']:
						usage='ignore'
					else:
						usage='Training'				 
					photo_name=os.path.basename(os.path.normpath(photo_file))
					pixels=str(image_from_pix(get_pix(photo_file)))
					pixels=pixels.replace(',','')
					pixels=pixels.replace(']','')
					pixels=pixels.replace('[','')
					pixels=pixels.replace('|','')
					stimwriter.writerow([emotion_dict[emotion_label],pixels,usage])

def direction_id_cropped(stimuli_folder,save_out):
	emotion_dict={'AN':0,'DI':1,'AF':2,'HA':3,'SA':4,'SU':5,'NE':6}
	id_folders=list(os.walk(stimuli_folder))
	with open(save_out, 'w') as csvfile:
		stimwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		stimwriter.writerow(['identity','pixels','Usage'])
		i_vals=np.asarray(range(0,len(id_folders)-1))
		for i in range(1,len(id_folders)):				
			identity_folder=id_folders[i][0]
			id_label=os.path.basename(os.path.normpath(identity_folder))
			add_factor=0
			if id_label[1] == 'M': #making sure we don't conflate M and F
				add_factor=35 #number of labels in Karolinska; may need to change w/ different dataset
			id_label=str(int(id_label[2:4])-1+add_factor)
			id_number=i-1
			for photo_file in os.listdir(identity_folder):
				emotion_label=photo_file[4:6]
				photo_file=os.path.join(identity_folder,photo_file)
				usage='Training'
				if photo_file.endswith("S.JPG"): #CHANGE THIS TO HOLD OUT A DIFFERENT VIEW
					usage='PublicTest'
				if photo_file.endswith("S.JPG") or photo_file.endswith("HL.JPG") or photo_file.endswith("HR.JPG"): #straight on, half left, half right 
					photo_name=os.path.basename(os.path.normpath(photo_file))
					pixels=str(image_from_pix(get_pix(photo_file)))
					pixels=pixels.replace(',','')
					pixels=pixels.replace(']','')
					pixels=pixels.replace('[','')
					pixels=pixels.replace('|','')
					stimwriter.writerow([id_label,pixels,usage])
					print(id_label)
					print(usage)

def direction_emotion_cropped(stimuli_folder,save_out):
	emotion_dict={'AN':0,'DI':1,'AF':2,'HA':3,'SA':4,'SU':5,'NE':6}
	id_folders=list(os.walk(stimuli_folder))
	with open(save_out, 'w') as csvfile:
		stimwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		stimwriter.writerow(['identity','pixels','Usage'])
		i_vals=np.asarray(range(0,len(id_folders)-1))
		for i in range(1,len(id_folders)):				
			identity_folder=id_folders[i][0]
			id_label=os.path.basename(os.path.normpath(identity_folder))
			add_factor=0
			if id_label[1] == 'M': #making sure we don't conflate M and F
				add_factor=35 #number of labels in Karolinska; may need to change w/ different dataset
			id_number=int(id_label[2:4])-1+add_factor
			for photo_file in os.listdir(identity_folder):
				photo_file=os.path.join(identity_folder,photo_file)
				usage='Training'
				if photo_file.endswith("S.JPG"): #CHANGE THIS TO HOLD OUT A DIFFERENT VIEW
					usage='PublicTest'
				if photo_file.endswith("S.JPG") or photo_file.endswith("HL.JPG") or photo_file.endswith("HR.JPG"): #straight on, half left, half right 
					photo_name=os.path.basename(os.path.normpath(photo_file))
					pixels=str(image_from_pix(get_pix(photo_file)))
					pixels=pixels.replace(',','')
					pixels=pixels.replace(']','')
					pixels=pixels.replace('[','')
					pixels=pixels.replace('|','')
					stimwriter.writerow([emotion_dict[emotion_label],pixels,usage])
					print(emotion_dict[emotion_label])
					print(id_number)
					print(usage)



stimuli_folder=r'/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/stimuli_karolinska'
#stimuli_folder=r'C:\Users\Katie\Desktop\SaxelabUROP\Karolinska\stimuli_karolinska';
#crop_out=r'/om/user/kco/saxelab/SaxelabUROP/stimuli_test_cropped_NEW_4.csv'
#crop_out=r'/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest/stimuli_test_cropped_NEW_5.csv'
#crop_out=r'C:\Users\Katie\Desktop\SaxelabUROP\Karolinska\test_crop_7.csv';
#big_out=r'C:\Users\Katie\Desktop\SaxelabUROP\Karolinska\stimuli_all.csv'
#emotion_out=r'C:\Users\Katie\Desktop\SaxelabUROP\Karolinska\stimuli_emotion.csv'
identity_out=r'/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest/identity_transfer_data/holdout_straight_id.csv'
#identity_out=r'C:\Users\Katie\Desktop\SaxelabUROP\Karolinska\holdout_sad_with45.csv'
#lower_out=r'C:\Users\Katie\Desktop\SaxelabUROP\Karolinska\stimuli_test.csv'

direction_id_cropped(stimuli_folder,identity_out)

