# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:33:05 2019

Rutika says that some minor modification was made to this code to be more
compatible with web server.

@author: jddel

CLIP_SINGLE_VIDEO (The key function to understand):
    - Purpose: Take a raw input video and output a video ready for use by the
      model.  Clip unneeded frames, crop unneeded areas, and rescale to desired
      dimensions.

    - Key Feature - Automatic frame clipping:  Since video recording starts
      when the user clicks on the record button and stop when he clicks the
      stop button, the videos will contain several frames of the user's hand
      and arm moving away from the mouse at the beginnning of the video and
      several frames of user moving back to click the mouse again at the end of
      the video.  Automatic frame clipping attempts to eliminate these
      extraneous frames and limit the video to just the frames of the sign
      being made.
        To do this, the average grayscale value by pixel is computed for some
      fraction of frames in the middle of the video
      (FRACTION_OF_FRAMES_FOR_REFERENCE).  These values are the reference values.
      Then, for each frame, for each pixel, the absolute delta between each
      pixel's value and its corresponding reference value are computed.  These
      are the "delta_to_ref" values.  For each frame, the mean of the
      delta_to_ref values is computed for pixels in the rows and columns
      defined by the DELTA_CROP_X_TUPLE and DELTA_CROP_Y_TUPLE tuples.  This
      metric measures how different pixels in the defined regions are for a frame
      compared to the reference group of frames.  For example, in the first few
      frames, we expect the user's hand is near his mouse, so his hand will be
      somewhere near the bottom of the screen (in the DELTA_CROP region, usually
      near the bottom of the video frame).  During the middle frames of the video,
      the user's hands are up higher, making a sign.  Therefore, the first and 
      last several frames will show up as having high delta_to_ref values, 
      because those frames include the user's hand near the mouse, while the 
      reference frames do not.
        Once we have the average delta_to_ref values in the "delta crop" region
      by frame, we can select frames to clip.  The function will exclude the
      first N frames whose delta_to_ref values exceed a certain fraction of the
      overall range (CUTOFF_FRACTION_OF_DELTA_RANGE), and the same is performed
      looking backwards from the last frame of the video, clipping frames until
      a frame is found whose delta_to_ref value is below the cutoff threshold.
        Additionally, more frames can be clipped after the last clipped frame
      at the start of the video and before the first clipped frame at the end
      of the video.  This allows clipping of frames that involve the user moving
      his arms from the mouse and into position to start the sign.  The additional
      frames to clip are set by the ADDITIONAL_LEADING_FRAMES_TO_CLIP and
      ADDITIONAL_TRAILING_FRAMES_TO_CLIP parameters.
        One more parameter, MIN_FRAMES_TO_KEEP, prevents over-clipping of
      videos.  If, the clipping algorithm would cause the kept number of frames
      to fall below MIN_FRAMES_TO_KEEP, then 1 additional frame on each end of
      the kept frames is added to the list of frames to keep until the number
      of frames is at or above MIN_FRAMES_TO_KEEP.

    - Cropping and scaling: The OUTPUT_CROP_X_TUPLE and OUTPUT_CROP_Y_TUPLE
      define the cropping to apply to the output video frames, and the
      OUTPUT_FRAME_XY_SIZE parameter (a tuple), defines the output frame dimensions to use.


IMPORTANT: USE THESE VALUES WHEN CALLING THE FUNCTIONS (THE MODEL WAS TRAINED
WITH VIDEOS BUILT USING THESE SETTINGS):

    DELTA_CROP_X_TUPLE=(0,0),
    DELTA_CROP_Y_TUPLE=(0,80),
    FRACTION_OF_FRAMES_FOR_REFERENCE=0.1,
    CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, # 0.3 worked well but sometimes was too aggressive
    ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, # At start of video, how many frames after last frame above threshold should be clipped
    ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10, # at end of video, how many frames before frames above threshold should be clipped
    MIN_FRAMES_TO_KEEP=20,
    OUTPUT_CROP_X_TUPLE=(130,130),
    OUTPUT_CROP_Y_TUPLE=(50,50),
    OUTPUT_FRAME_XY_SIZE=(299,299),
    OUTPUT_FRAMES_PER_SECOND=30.0,

CLIP_ALL_VIDEOS_IN_DIR - runs CLIP_SINGLE_VIDEO for all video files in a provided input directory
CLIP_LIST_OF_VIDEOS -  runs CLIP_SINGLE_VIDEO for all video files listed in a provided dataframe (must include columns: Sign, VIDEO_FILE, and OUTPUT_FILE. )
"""

# For preparing images
import cv2
from skimage import io
from skimage import color
from skimage import transform
from skimage.util import crop
import os

from skimage import color

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ntpath


def CLIP_SINGLE_VIDEO(
        VIDEO_FILE, #filename of input video file
        OUTPUT_FILE, # filename of output video file
        FOURCC='XVID', # fourcc encoding to use in output video
        FRAMES_TO_CLIP_TUPLE=None, # N frames to clip from beginning and end of video.  Entering a tuple here overrides the automatic clipping entirely
        DELTA_CROP_X_TUPLE=(0,0), # N pixels to crop from left and right
        DELTA_CROP_Y_TUPLE=(0,80), # N pixels 
        FRACTION_OF_FRAMES_FOR_REFERENCE=0.1, # Fraction of frames to use to compute reference pixel values (from which delta will be computed)
        CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, # Frames whose mean 
        ADDITIONAL_LEADING_FRAMES_TO_CLIP=0, # At start of video, how many frames after last frame above threshold should be clipped
        ADDITIONAL_TRAILING_FRAMES_TO_CLIP=0, # at end of video, how many frames before frames above threshold should be clipped
        MIN_FRAMES_TO_KEEP=20, # If computed N frames to keep is below this number, additional frames are added to beginning and to end of video until this number is met.
        OUTPUT_CROP_X_TUPLE=(130,130), # Left and right cropping to apply to the generated video
        OUTPUT_CROP_Y_TUPLE=(50,50), # top and bottom cropping to apply to the generated video
        OUTPUT_FRAME_XY_SIZE=(299,299), # Final size for frames of output video
        OUTPUT_FRAMES_PER_SECOND=30.0, # Frames per second to use for generated output video
        DELTA_TO_REF_VIDEO_OUTPUT_DIR=None # path to where "delta to reference" videos should be written
        ):

#    VIDEO_FILE= r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\BATCH2_jim_recorded_signs\1563816900504.webm'
#    OUTPUT_FILE=r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\BATCH2_jim_recorded_signs\BATCH2_CLIPPED_VIDEOS\1563816900504.avi'
#    FOURCC='XVID'
#    FRAMES_TO_CLIP_TUPLE=None
#    DELTA_CROP_X_TUPLE=(0,0)
#    DELTA_CROP_Y_TUPLE=(0,80)
#    FRACTION_OF_FRAMES_FOR_REFERENCE=0.1
#    CUTOFF_FRACTION_OF_DELTA_RANGE=0.3
#    ADDITIONAL_LEADING_FRAMES_TO_CLIP=10 # At start of video, how many frames after last frame above threshold should be clipped
#    ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10 # at end of video, how many frames before frames above threshold should be clipped
#    OUTPUT_CROP_X_TUPLE=(130,130)
#    OUTPUT_CROP_Y_TUPLE=(50,50)
#    OUTPUT_FRAME_XY_SIZE=(299,299)
#    OUTPUT_FRAMES_PER_SECOND=30.0
#    DELTA_TO_REF_VIDEO_OUTPUT_DIR=r'c:\temp' + '\\'

    # Make the output dir if it doesn't exist
    output_dir=OUTPUT_FILE[:-len(ntpath.basename(OUTPUT_FILE))-1]
    if (output_dir != '' and not(os.path.exists(output_dir))):
        os.makedirs(output_dir)

    # read video file
    #print("about to read file: " + VIDEO_FILE)
    cap = cv2.VideoCapture(VIDEO_FILE)
    success, image = cap.read()
    list_of_all_frames=[]
    list_of_all_frames_grayscale=[]
    count = 0
    while success:
        list_of_all_frames.append(image)
        list_of_all_frames_grayscale.append(color.rgb2gray(image))
        # read the next frame
        success, image = cap.read()
        count += 1
    cap.release()
    # create the videos
    full_video=np.array(list_of_all_frames)
    full_video_grayscale=np.array(list_of_all_frames_grayscale)
    n_frames, height, width, channels = full_video.shape

    #### If we are using user-provided clipping, just user user defined frame clip values.    
    if FRAMES_TO_CLIP_TUPLE is not None:
        first_frame_to_keep=FRAMES_TO_CLIP_TUPLE[0]
        last_frame_to_keep=len(full_video)-FRAMES_TO_CLIP_TUPLE[1]-1
    ### If we're smart, we'll do intelligent clipping based upon activity in the crop region
    else:
        #### Get reference pixel values ###############
        # 1. Get the mean value by pixel for the middle 20% of frames.  This is the time
        # when the hands are making the sign.  Hands are out of the crop zone, so we
        # can see the baseline values of the crop zone, call it "baseline"
        ref_start=int(n_frames*(0.5-FRACTION_OF_FRAMES_FOR_REFERENCE/2))
        ref_end=int(n_frames*(0.5+FRACTION_OF_FRAMES_FOR_REFERENCE/2))

        # this returns the average pixel value across frames for the frames of interest (why the +1 ?)        
        ref_pixel_values=np.average(full_video_grayscale[ref_start:ref_end+1], axis=0)
        #print("Reference frames: " + str(ref_start) + " - " + str(ref_end))        
        # 2. For each frame, for each pixel, compute the abs delta to baseline.
        delta_to_ref=np.absolute(full_video_grayscale - ref_pixel_values)
        delta_to_ref=(delta_to_ref*255).astype(np.uint8)

        ## Here is where we can make a video of absdelta (only done if path provided)
        if DELTA_TO_REF_VIDEO_OUTPUT_DIR is not None:
            if not(os.path.exists(DELTA_TO_REF_VIDEO_OUTPUT_DIR)):
                os.makedirs(DELTA_TO_REF_VIDEO_OUTPUT_DIR)
            output_filename=DELTA_TO_REF_VIDEO_OUTPUT_DIR + 'DELTA_TO_REF_' + ntpath.basename(OUTPUT_FILE)
            # add white crop lines
            delta_to_ref_marked=delta_to_ref.copy()
            delta_to_ref_marked[:,DELTA_CROP_Y_TUPLE[0],:]=255
            delta_to_ref_marked[:,-DELTA_CROP_Y_TUPLE[1],:]=255
            delta_to_ref_marked[:,:,DELTA_CROP_X_TUPLE[0]]=255
            delta_to_ref_marked[:,:,-DELTA_CROP_X_TUPLE[1]]=255
            out = cv2.VideoWriter(output_filename,
                                  cv2.VideoWriter_fourcc(FOURCC[0],FOURCC[1],FOURCC[2],FOURCC[3]), float(OUTPUT_FRAMES_PER_SECOND), (width,height), 0)
            for i in range(len(delta_to_ref_marked)):
                out.write(delta_to_ref_marked[i])
            out.release()
            # io.imshow(delta_to_ref_marked[20])
            # io.imshow(delta_to_ref_marked[27])
            # io.imshow(cv2.cvtColor(full_video[30], cv2.COLOR_BGR2RGB))
    
        
        # 3. For each frame, across all crop zone pixels, compute mean and max absdelta
        # to baseline.  We expect this absdelta to be high for the firts few frames, 
        # then low for middle frames, then high again as the user clicks off the recording.
        # The shape of data here will be 1 number (per stat) per frame.
    
        # compute number of pixels in crop zone
        n_cropped_pixels=sum(DELTA_CROP_Y_TUPLE)*width + (height-sum(DELTA_CROP_Y_TUPLE))*sum(DELTA_CROP_X_TUPLE)  
        # seed sum and max
        mean_deltas=[]
        max_deltas=[]
        for curr_frame in delta_to_ref:
            # curr_frame =delta_to_ref[0]
            sum_delta=0
            max_delta=0
            # the top cropped section (all cols)
            if DELTA_CROP_Y_TUPLE[0]>0:
                sum_delta+=np.sum(curr_frame[:DELTA_CROP_Y_TUPLE[0],])
                max_delta=max(max_delta, np.max(curr_frame[:DELTA_CROP_Y_TUPLE[0],]))
            # the LEFT cropped columns of the non-cropped rows
            if DELTA_CROP_X_TUPLE[0]>0:
                sum_delta+=np.sum(curr_frame[DELTA_CROP_Y_TUPLE[0]:height-DELTA_CROP_Y_TUPLE[1],:DELTA_CROP_X_TUPLE[0]])
                max_delta=max(max_delta, np.max(curr_frame[DELTA_CROP_Y_TUPLE[0]:height-DELTA_CROP_Y_TUPLE[1],:DELTA_CROP_X_TUPLE[0]]))
            # the RIGHT cropped columns of the non-cropped rows
            if DELTA_CROP_X_TUPLE[1]>0:
                sum_delta+=np.sum(curr_frame[DELTA_CROP_Y_TUPLE[0]:height-DELTA_CROP_Y_TUPLE[1],-DELTA_CROP_X_TUPLE[1]:])
                max_delta=max(max_delta,np.max(curr_frame[DELTA_CROP_Y_TUPLE[0]:height-DELTA_CROP_Y_TUPLE[1],-DELTA_CROP_X_TUPLE[1]:]))
            # the BOTTOM cropped section (all cols)
            if DELTA_CROP_Y_TUPLE[1]>0:
                sum_delta+=np.sum(curr_frame[-DELTA_CROP_Y_TUPLE[1]:,]) 
                max_delta=max(max_delta,np.max(curr_frame[-DELTA_CROP_Y_TUPLE[1]:,]))


            mean_deltas.append(sum_delta/n_cropped_pixels)
            max_deltas.append(max_delta)
    
    
    
        # 4. Cut off all beginning frames before the stat falls below a threshold, and
        # cut off all ending frames on/after the stat rises above the threshold.
        ### we will clip all leading frames where ave_delta is > 50% of the range of the ave_deltas
        range_of_mean_deltas=max(mean_deltas)-min(mean_deltas)
        max_allowable_delta=min(mean_deltas)+CUTOFF_FRACTION_OF_DELTA_RANGE*range_of_mean_deltas
        ### Clip leading frames ###
        nth_frame=0
        while mean_deltas[nth_frame] > max_allowable_delta:
            nth_frame+=1
        first_frame_to_keep=nth_frame+ADDITIONAL_LEADING_FRAMES_TO_CLIP
        ### Clip trailing frames ###
        nth_frame=len(mean_deltas)-1
        while mean_deltas[nth_frame] > max_allowable_delta:
            nth_frame-=1
        last_frame_to_keep=nth_frame-ADDITIONAL_TRAILING_FRAMES_TO_CLIP
        
        # safety! don't go below MIN_FRAMES_TO_KEEP
        minstring=""
        while (last_frame_to_keep-first_frame_to_keep < MIN_FRAMES_TO_KEEP) and ((first_frame_to_keep>0) or (last_frame_to_keep<len(full_video)-1)):
            minstring="Force Including additional frames to keep at least " + str(MIN_FRAMES_TO_KEEP) + " frames in final output.\n"
            first_frame_to_keep=max(0,first_frame_to_keep-1)
            last_frame_to_keep=min(len(full_video)-1,last_frame_to_keep+1)
        
        # Also, if I'm outputting the delta video, I probably want to see the plot
        if DELTA_TO_REF_VIDEO_OUTPUT_DIR is not None:
            ddf=pd.DataFrame({'frame':[x for x in range(len(mean_deltas))],'mean':mean_deltas,'max':max_deltas,
                                       'frac_of_range':[(x-min(mean_deltas))/range_of_mean_deltas for x in mean_deltas]})
            #ddf.to_csv(r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\framedeltas_10ref.csv')
            plt.plot(ddf['frame'], ddf['frac_of_range'], 'o', color='black')
    ##### END: else, we are doing automatic clip selection ####################


    ## Write the clipped video! ###

    # Now it's time to apply the output_cropping
    cropped_video=full_video[:,OUTPUT_CROP_Y_TUPLE[0]:height-OUTPUT_CROP_Y_TUPLE[1],OUTPUT_CROP_X_TUPLE[0]:width-OUTPUT_CROP_X_TUPLE[1]]
    
    ### Create the videowriter object with the final size
    out = cv2.VideoWriter(OUTPUT_FILE,cv2.VideoWriter_fourcc(FOURCC[0],FOURCC[1],FOURCC[2],FOURCC[3]), float(OUTPUT_FRAMES_PER_SECOND), OUTPUT_FRAME_XY_SIZE)
    for i in range(first_frame_to_keep,last_frame_to_keep+1):
        # resize the frame  i=0
        resized_frame=cv2.resize(cropped_video[i], OUTPUT_FRAME_XY_SIZE, interpolation =cv2.INTER_AREA)
        # writing to a image array
        out.write(resized_frame)
    out.release()
    print("------------------------------\n" \
          + "Input Video: " + VIDEO_FILE + "\n" \
          + "Output Video: " + OUTPUT_FILE + "\n" \
          + "Input N_Frames: " + str(len(full_video)) + "\n" \
          + "Output N_Frames: " + str(last_frame_to_keep - first_frame_to_keep + 1) + "\n" \
          + minstring \
          + "First, Last Kept Frames (0-based): " + str(first_frame_to_keep) + ", " + str(last_frame_to_keep) + "\n" \
          + "Frames Clipped Start/End: " + str(first_frame_to_keep) + ", " + str(len(full_video) - last_frame_to_keep -1)  + "\n"\
          + "------------------------------\n")
    
############### END FUNCTION: CLIP_SINGLE_VIDEO ################################


def CLIP_ALL_VIDEOS_IN_DIR(
        INPUT_DIR,
        OUTPUT_DIR,
        INPUT_EXTENSION='webm',
        OUTPUT_EXTENSION='avi',
        FOURCC='XVID',
        FRAMES_TO_CLIP_TUPLE=None,
        MAX_FILES_TO_PROCESS=None,
        DELTA_CROP_X_TUPLE=(0,0),
        DELTA_CROP_Y_TUPLE=(0,80),
        FRACTION_OF_FRAMES_FOR_REFERENCE=0.1,
        CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, # 0.3 worked well but sometimes was too aggressive
        ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, # At start of video, how many frames after last frame above threshold should be clipped
        ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10, # at end of video, how many frames before frames above threshold should be clipped
        MIN_FRAMES_TO_KEEP=20,
        OUTPUT_CROP_X_TUPLE=(130,130),
        OUTPUT_CROP_Y_TUPLE=(50,50),
        OUTPUT_FRAME_XY_SIZE=(299,299),
        OUTPUT_FRAMES_PER_SECOND=30.0,
        DELTA_TO_REF_VIDEO_OUTPUT_DIR=None
        ):
#    INPUT_DIR=r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\jim_recorded_signs' + '\\'
#    OUTPUT_DIR=r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\jim_recorded_signs\CLIPPED_VIDEOS' + '\\'
#    INPUT_EXTENSION='webm'
#    OUTPUT_EXTENSION='avi'
#    MAX_FILES_TO_PROCESS=5
#    FOURCC='MJPG'
#    FRAMES_TO_CLIP_TUPLE=None
#    DELTA_CROP_X_TUPLE=(163,178)
#    DELTA_CROP_Y_TUPLE=(60,121)
#    FRACTION_OF_FRAMES_FOR_REFERENCE=0.1
#    CUTOFF_FRACTION_OF_DELTA_RANGE=0.5
#    OUTPUT_FRAMES_PER_SECOND=20.0
#    DELTA_TO_REF_VIDEO_OUTPUT_DIR=None
    
    # get files to process
    files_to_process=os.listdir(INPUT_DIR)
    files_to_process=[f for f in files_to_process if f.endswith('.'+INPUT_EXTENSION)]
    # if max files provided, limit list to that amount
    if MAX_FILES_TO_PROCESS is not None:
        files_to_process=files_to_process[:MAX_FILES_TO_PROCESS]


    # Make the output dir if it doesn't exist
    if not(os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)


    ## Loop through all videos!
    i=0
    for v in files_to_process:
        i+=1
        print("\n=============== Processing File: " + v + " (" + str(i) + " of " + str(len(files_to_process)) + ") ===============")
        #v=files_to_process[0]
        curr_output_file= OUTPUT_DIR + ntpath.basename(v)[:ntpath.basename(v).rfind('.')]+'.'+OUTPUT_EXTENSION
        CLIP_SINGLE_VIDEO(
                VIDEO_FILE=INPUT_DIR + v,
                OUTPUT_FILE=curr_output_file,
                FOURCC=FOURCC,
                FRAMES_TO_CLIP_TUPLE=FRAMES_TO_CLIP_TUPLE,
                DELTA_CROP_X_TUPLE=DELTA_CROP_X_TUPLE,
                DELTA_CROP_Y_TUPLE=DELTA_CROP_Y_TUPLE,
                FRACTION_OF_FRAMES_FOR_REFERENCE=FRACTION_OF_FRAMES_FOR_REFERENCE,
                CUTOFF_FRACTION_OF_DELTA_RANGE=CUTOFF_FRACTION_OF_DELTA_RANGE,
                ADDITIONAL_LEADING_FRAMES_TO_CLIP=ADDITIONAL_LEADING_FRAMES_TO_CLIP, # At start of video, how many frames after last frame above threshold should be clipped
                ADDITIONAL_TRAILING_FRAMES_TO_CLIP=ADDITIONAL_TRAILING_FRAMES_TO_CLIP, # at end of video, how many frames before frames above threshold should be clipped
                MIN_FRAMES_TO_KEEP=MIN_FRAMES_TO_KEEP,
                OUTPUT_CROP_X_TUPLE=OUTPUT_CROP_X_TUPLE,
                OUTPUT_CROP_Y_TUPLE=OUTPUT_CROP_Y_TUPLE,
                OUTPUT_FRAME_XY_SIZE=OUTPUT_FRAME_XY_SIZE,
                OUTPUT_FRAMES_PER_SECOND=OUTPUT_FRAMES_PER_SECOND,
                DELTA_TO_REF_VIDEO_OUTPUT_DIR=DELTA_TO_REF_VIDEO_OUTPUT_DIR
                )
    # end for loop
    print("All done processing all videos!")
######### End function: CLIP_ALL_VIDEOS_IN_DIR ################################
        


    


#CLIP_ALL_VIDEOS_IN_DIR(
#        INPUT_DIR=r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\jim_recorded_signs' + '\\',
#        OUTPUT_DIR=r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\jim_recorded_signs\CLIPPED_VIDEOS_50' + '\\',
#        INPUT_EXTENSION='webm',
#        OUTPUT_EXTENSION='avi',
#        FOURCC='XVID',
#        MAX_FILES_TO_PROCESS=None,
#        FRAMES_TO_CLIP_TUPLE=None,
#        DELTA_CROP_X_TUPLE=(0,0),
#        DELTA_CROP_Y_TUPLE=(0,80),
#        FRACTION_OF_FRAMES_FOR_REFERENCE=0.1,
#        CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, #<---- THIS TUNES HOW SENSITIVE THE CLIPPING IS. LOWER=MORE FRAMES REMOVED
#        ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, # At start of video, how many frames after last frame above threshold should be clipped
#        ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10, # at end of video, how many frames before frames above threshold should be clipped
#        OUTPUT_CROP_X_TUPLE=(130,130),
#        OUTPUT_CROP_Y_TUPLE=(50,50),
#        OUTPUT_FRAME_XY_SIZE=(299,299),
#        OUTPUT_FRAMES_PER_SECOND=30.0,
#        DELTA_TO_REF_VIDEO_OUTPUT_DIR=None
#        )





def CLIP_LIST_OF_VIDEOS(
        VIDEO_LIST_DF,
        FOURCC='XVID',
        MAX_FILES_TO_PROCESS=None,
        FRAMES_TO_CLIP_TUPLE=None,
        DELTA_CROP_X_TUPLE=(0,0),
        DELTA_CROP_Y_TUPLE=(0,80),
        FRACTION_OF_FRAMES_FOR_REFERENCE=0.1,
        CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, #<---- THIS TUNES HOW SENSITIVE THE CLIPPING IS. LOWER=MORE FRAMES REMOVED
        ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, # At start of video, how many frames after last frame above threshold should be clipped
        ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10, # at end of video, how many frames before frames above threshold should be clipped
        MIN_FRAMES_TO_KEEP=20,
        OUTPUT_CROP_X_TUPLE=(130,130),
        OUTPUT_CROP_Y_TUPLE=(50,50),
        OUTPUT_FRAME_XY_SIZE=(299,299),
        OUTPUT_FRAMES_PER_SECOND=30.0,
        DELTA_TO_REF_VIDEO_OUTPUT_DIR=None
        ):
    
    # get len
    nvideos=len(VIDEO_LIST_DF)
    nth_video=0
    # loop through VIDEO_LIST_DF
    for index, row in VIDEO_LIST_DF.iterrows():
        nth_video+=1
        print("CLIP_LIST_OF_VIDEOS: Processing video " + str(nth_video) + " of " + str(nvideos) + ". Sign=" + row['Sign'])
        CLIP_SINGLE_VIDEO(
            VIDEO_FILE=row['VIDEO_FILE'],
            OUTPUT_FILE=row['OUTPUT_FILE'],
            FOURCC=FOURCC,
            FRAMES_TO_CLIP_TUPLE=FRAMES_TO_CLIP_TUPLE,
            DELTA_CROP_X_TUPLE=DELTA_CROP_X_TUPLE,
            DELTA_CROP_Y_TUPLE=DELTA_CROP_Y_TUPLE,
            FRACTION_OF_FRAMES_FOR_REFERENCE=FRACTION_OF_FRAMES_FOR_REFERENCE,
            CUTOFF_FRACTION_OF_DELTA_RANGE=CUTOFF_FRACTION_OF_DELTA_RANGE,
            ADDITIONAL_LEADING_FRAMES_TO_CLIP=ADDITIONAL_LEADING_FRAMES_TO_CLIP, # At start of video, how many frames after last frame above threshold should be clipped
            ADDITIONAL_TRAILING_FRAMES_TO_CLIP=ADDITIONAL_TRAILING_FRAMES_TO_CLIP, # at end of video, how many frames before frames above threshold should be clipped
            MIN_FRAMES_TO_KEEP=MIN_FRAMES_TO_KEEP,
            OUTPUT_CROP_X_TUPLE=OUTPUT_CROP_X_TUPLE,
            OUTPUT_CROP_Y_TUPLE=OUTPUT_CROP_Y_TUPLE,
            OUTPUT_FRAME_XY_SIZE=OUTPUT_FRAME_XY_SIZE,
            OUTPUT_FRAMES_PER_SECOND=OUTPUT_FRAMES_PER_SECOND,
            DELTA_TO_REF_VIDEO_OUTPUT_DIR=DELTA_TO_REF_VIDEO_OUTPUT_DIR
        )
### end function CLIP_LIST_OF_VIDEOS #######################################
        
        















### The code below prepares files for Chandan ################################################################
### The code below prepares files for Chandan ################################################################
### The code below prepares files for Chandan ################################################################
### The code below prepares files for Chandan ################################################################
### The code below prepares files for Chandan ################################################################
### The code below prepares files for Chandan ################################################################
        


# First I'll do the round 2 videos (I was more methodical in methods)

#vdf=pd.read_excel(r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\signs recorded.xlsx', sheet_name='round_2_file_list')
#vdf1=vdf[pd.notnull(vdf['chandan'])]
#again=vdf1[vdf1['Sign']=='AGAIN']
#notagain=vdf1[vdf1['Sign']!='AGAIN']
#        
#CLIP_LIST_OF_VIDEOS(
#        VIDEO_LIST_DF=notagain,
#        FOURCC='XVID',
#        MAX_FILES_TO_PROCESS=None,
#        FRAMES_TO_CLIP_TUPLE=None,
#        DELTA_CROP_X_TUPLE=(0,0),
#        DELTA_CROP_Y_TUPLE=(0,80),
#        FRACTION_OF_FRAMES_FOR_REFERENCE=0.1,
#        CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, #<---- THIS TUNES HOW SENSITIVE THE CLIPPING IS. LOWER=MORE FRAMES REMOVED
#        ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, # At start of video, how many frames after last frame above threshold should be clipped
#        ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10, # at end of video, how many frames before frames above threshold should be clipped
#        MIN_FRAMES_TO_KEEP=20,
#        OUTPUT_CROP_X_TUPLE=(130,130),
#        OUTPUT_CROP_Y_TUPLE=(50,50),
#        OUTPUT_FRAME_XY_SIZE=(299,299),
#        OUTPUT_FRAMES_PER_SECOND=30.0,
#        DELTA_TO_REF_VIDEO_OUTPUT_DIR=None
#        )        
#
#nonchandan=set(vdf['Sign']) - set(vdf1['Sign'])
#
#notchandan=vdf[vdf['Sign'].isin(nonchandan)]
#
#        
#CLIP_LIST_OF_VIDEOS(
#        VIDEO_LIST_DF=notchandan,
#        FOURCC='XVID',
#        MAX_FILES_TO_PROCESS=None,
#        FRAMES_TO_CLIP_TUPLE=None,
#        DELTA_CROP_X_TUPLE=(0,0),
#        DELTA_CROP_Y_TUPLE=(0,80),
#        FRACTION_OF_FRAMES_FOR_REFERENCE=0.1,
#        CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, #<---- THIS TUNES HOW SENSITIVE THE CLIPPING IS. LOWER=MORE FRAMES REMOVED
#        ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, # At start of video, how many frames after last frame above threshold should be clipped
#        ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10, # at end of video, how many frames before frames above threshold should be clipped
#        MIN_FRAMES_TO_KEEP=20,
#        OUTPUT_CROP_X_TUPLE=(130,130),
#        OUTPUT_CROP_Y_TUPLE=(50,50),
#        OUTPUT_FRAME_XY_SIZE=(299,299),
#        OUTPUT_FRAMES_PER_SECOND=30.0,
#        DELTA_TO_REF_VIDEO_OUTPUT_DIR=None
#        )        





############# Now I'll go ahead and try conversion in my Round 1 videos.
#vdf_round1=pd.read_excel(r'F:\users\jddel\Documents\DATA_SCIENCE_DEGREE_LAPTOP\W210_CAPSTONE\signs recorded.xlsx', sheet_name='round_1_file_list')
#vdf1_round1=vdf_round1[vdf_round1['Batch']==1]
#set(vdf1_round1['Sign'])
#
#CLIP_LIST_OF_VIDEOS(
#        VIDEO_LIST_DF=vdf1_round1,
#        FOURCC='XVID',
#        MAX_FILES_TO_PROCESS=None,
#        FRAMES_TO_CLIP_TUPLE=None,
#        DELTA_CROP_X_TUPLE=(0,0),
#        DELTA_CROP_Y_TUPLE=(0,80),
#        FRACTION_OF_FRAMES_FOR_REFERENCE=0.1,
#        CUTOFF_FRACTION_OF_DELTA_RANGE=0.4, #<---- THIS TUNES HOW SENSITIVE THE CLIPPING IS. LOWER=MORE FRAMES REMOVED
#        ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, # At start of video, how many frames after last frame above threshold should be clipped
#        ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10, # at end of video, how many frames before frames above threshold should be clipped
#        MIN_FRAMES_TO_KEEP=20,
#        OUTPUT_CROP_X_TUPLE=(130,130),
#        OUTPUT_CROP_Y_TUPLE=(50,50),
#        OUTPUT_FRAME_XY_SIZE=(299,299),
#        OUTPUT_FRAMES_PER_SECOND=30.0,
#        DELTA_TO_REF_VIDEO_OUTPUT_DIR=None
#        )        

