import pandas as pd
import os
import cv2

# %%

dataset_root = 'VIRAT Ground Dataset/'
annotations_dir = dataset_root + 'annotations/'
annotations_files = os.listdir(annotations_dir)

event_features = ['eventID', 'event_type', 'duration', 'start_frame', 'end_frame', 'current_frame', 'bbox_lefttop x',
                  'bbox_lefttop y', 'bbox_width', 'bbox_height']
object_features = ['Object id','object_duration','current_frame', 'bbox_lefttop x',
                   'bbox_lefttop y', 'bbox_width', 'bbox_height','object_type']
object_type = 1 # person
event_type = 11 # person entering a facility


#%%
for annotations_file in annotations_files:
	try:
		if annotations_file.split('.')[2] == 'events':
			test_file = annotations_dir + annotations_file
			object_file = annotations_file.split('.')[0] +'.viratdata.objects.txt'
			object_file = annotations_dir + object_file
			object_df = pd.read_csv(object_file, delimiter=' ', names=object_features, index_col=False)
			object_df = object_df[object_df['object_type']==object_type]



			event_df = pd.read_csv(test_file, delimiter=' ', names=event_features, index_col=False)
			if (event_df['event_type'] == event_type                                                                                                                                                                                                                                                                                                                                                                        ).sum() > 0:
				print(test_file)
				video_file_name = annotations_file.split('.')[0] + '.mp4'
				print(video_file_name)
				event_df = event_df[event_df['event_type'] == event_type]

				start_frame = 0
				end_frame = 0
				video_path = dataset_root + 'videos_original/' + video_file_name
				cap = cv2.VideoCapture(video_path)
				for index, row in event_df.iterrows():
					start_frame = row['start_frame']
					end_frame = row['end_frame']
					current_frame = row['current_frame']
					current_frame_objects = []
					frame_filtered_objects = object_df[object_df['current_frame']==current_frame]
					if start_frame == current_frame:
						cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

					frame_read, frame = cap.read()

					pt1 = (row['bbox_lefttop x'], row['bbox_lefttop y'])
					pt2 = (pt1[0] + row['bbox_width'], pt1[1] + row['bbox_height'])
					cv2.rectangle(frame, pt1, pt2, (255, 255, 255))
					for object_index, object_row in frame_filtered_objects.iterrows():
						pt1 = (object_row['bbox_lefttop x'], object_row['bbox_lefttop y'])
						pt2 = (pt1[0] + object_row['bbox_width'], pt1[1] + object_row['bbox_height'])
						cv2.rectangle(frame, pt1, pt2, (0, 255, 0))

					cv2.imshow("vid", frame)
					cv2.resizeWindow('vid',1280,720)
					cv2.waitKey(30)

					start_frame += 1
				cv2.destroyWindow('vid')
	except:
		print('Error in: ')
		print(annotations_file)
		print('')
		continue
