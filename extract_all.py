import os

import cv2
import pandas as pd


# %%
def mkdir(dir):
	try:
		os.mkdir(dir)
	except:
		print('Cant create ', dir)


dataset_root = 'VIRAT Ground Dataset/'
annotations_dir = dataset_root + 'annotations/'
annotations_files = os.listdir(annotations_dir)

event_features = ['eventID', 'event_type', 'duration', 'start_frame', 'end_frame', 'current_frame', 'bbox_lefttop x',
                  'bbox_lefttop y', 'bbox_width', 'bbox_height']
object_features = ['Object id', 'object_duration', 'current_frame', 'bbox_lefttop x',
                   'bbox_lefttop y', 'bbox_width', 'bbox_height', 'object_type']
object_type = 1  # person
total_events = 11  # person entering a facility

# %%
events = ['1: Person loading an Object to a Vehicle',
          '2: Person Unloading an Object from a Car/Vehicle',
          '3: Person Opening a Vehicle/Car Trunk',
          '4: Person Closing a Vehicle/Car Trunk',
          '5: Person getting into a Vehicle',
          '6: Person getting out of a Vehicle',
          '7: Person gesturing',
          '8: Person digging',
          '9: Person carrying an object',
          '10: Person running',
          '11: Person entering a facility',
          '12: Person exiting a facility']
extraction_dir = 'image_extractions/'
mkdir(extraction_dir)
videos_dir = dataset_root + 'videos_original/'
frame_skip = 10
for video_name in os.listdir(videos_dir):

	video_path = videos_dir + video_name
	print('Processing ',video_path)
	try:
		current_frame = 0
		cap = cv2.VideoCapture(video_path)
		object_file_path = annotations_dir + video_name.split('.')[0] + '.viratdata.objects.txt'
		object_df = pd.read_csv(object_file_path, delimiter=' ', names=object_features, index_col=False)
		object_df = object_df[object_df['object_type'] == object_type]
		while True:
			ret, frame = cap.read()
			if not ret:
				cap.release()
				break

			frame_filtered_objects = object_df[object_df['current_frame'] == current_frame]
			for object_index, object_row in frame_filtered_objects.iterrows():
				pt1 = (object_row['bbox_lefttop x'], object_row['bbox_lefttop y'])
				pt2 = (pt1[0] + object_row['bbox_width'], pt1[1] + object_row['bbox_height'])
				image_name = video_name.split('.')[0] + '.' + str(object_row['Object id']) + '.' + str(
					current_frame) + '.png'
				cv2.imwrite(extraction_dir + image_name, frame[pt1[1]:pt2[1], pt1[0]:pt2[0]])
			# cv2.rectangle(frame, pt1, pt2, (0, 255, 0))
			# cv2.imshow("vid", frame)
			# cv2.resizeWindow('vid', 600, 600)
			# cv2.waitKey(30)


			total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

			print(int(current_frame/total_frames*100) ,' percent complete')
			current_frame = current_frame + frame_skip + 1

			if frame_skip>0:
				cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame)
	except:
		print('Error processing ', video_path)
