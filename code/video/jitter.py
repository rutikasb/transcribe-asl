import numpy as np
import random
from statistics import median

# author: James De La Torre

class Jitter:
    def get_strobed_videos(
            framedata,
            n_frames_to_keep,
            class_label=None,
            jitter=False,
            n_jittered_to_create=100,
            unique_results_only=True):
        """
        Takes as input a numpy array, framedata (e.g., a SINGLE VIDEO, or an array of filenames of frames for a single video).
        Returns an array that is an ARRAY OF "STROBED" VIDEOS (keeping only a sample of frames), even if
        only 1 video worth of data is returned.

        If the jitter option is set to True, the function will generate multiple sequences of data by randomly selecting
        a frame from each "frame group".

        Example: Say we have 50 total frames and want to keep 10.  This means we will keep 1 out of every 5 frames.
        The frames are grouped into 10 "frame groups", each containing 50/10=5 frames.  The group number by frame will look like
        group:[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, . . . , 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]
        index:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, . . . ,40,41,42,43,44,45,46,47,48,49]
        Without jitter, we keep the middle frame from each group: [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
        With jitter, a random frame is selected from each group (1 frame from group 0, 1 from group 1, etc.)
        The n_jittered_to_create option tells the function how many jittered sequences to create.
        If the unique_results_only flag is True, then only the unique generated jittered sequences are kept
        (exact duplicates are removed).

        The class_label, if provided, must be an integer.  If it is provided, the function will
        also return a 1d numpy array whose values are all the class_label (this is useful when making jittered sequences
        so you have class labels for all your generated records.)


        """

        # Safety check class_label must be none or int
        if (class_label is not None) and type(class_label) != int:
            raise Exception('class_label must either be None or an integer.')
            return None


        # Count the frames in the data and get height and width
        nframes = framedata.shape[0]
        #nframes, height, width = framedata.shape

        ### safety check.  Do we have enough frames?
        #if n_frames_to_keep > nframes:
        #    raise Exception('Number of frames requested (' + str(n_frames_to_keep) + ') exceeds number of frames in raw data (' + str(nframes) + ')')
        #    return None

        # Degenerate case: we request more frames that we have.
        if n_frames_to_keep >= nframes:
            print('Number of frames requested (' + str(n_frames_to_keep) + ') equals or exceeds number of frames in raw data (' \
                  + str(nframes) + ').  Will return all frames.  Jittering is not possible.')
            if class_label is not None:
                return framedata, np.array([class_label])
            else:
                return framedata



        # get a nice list of indices
        findices=[i for i in range(nframes)]
        # Fraction of frames that will be kept
        frac_to_keep=n_frames_to_keep/nframes
        print("Input frame data has " + str(nframes) + " frames.  User requested to keep " + str(n_frames_to_keep) + " frames, which is " + str(round(frac_to_keep*100,1)) + "% of original frame count.")

        # nth group of frames from which to choose
        fgroups=[int(round(float(i)*frac_to_keep,2)) for i in range(nframes)]
        #get count of frames in each frame group and make a dict of them
        fgroupcounts=[(g,fgroups.count(g)) for g in set(fgroups)]
        gcdict=dict(fgroupcounts)


        ####### Now get list of lists of indices we want to keep ####################

        # start with nothing
        all_sequences_to_keep=[]

        # IFF we jitter, we will output multiple strobed videos.
        if jitter:
            for curr_sequence in range(n_jittered_to_create):
                # start with no frame indices to keep
                frame_indices_to_keep=[]
                # for each index group (set of candidate frames for the nth strobed frame)
                for igroup in list(gcdict.keys()):
                    # Get a mask that selects just indices corresponding to the current frame group
                    curr_mask=[v==igroup for v in fgroups]
                    # get the indices
                    curr_subset=[i for i, j in zip(findices, curr_mask) if j]
                    # choose 1 at random
                    chosen=random.choice(curr_subset)
                    # and add it to the list of selected ("strobed") frames
                    frame_indices_to_keep.append(chosen)
                # once we've gone through all the index groups and made our sequence, add it to all_sequences_to_keep
                all_sequences_to_keep.append(frame_indices_to_keep)
            # Now optionally ensure that we didn't randomly create same sequence more than once
            if unique_results_only:
                print("Before de-duping, there are " + str(len(all_sequences_to_keep)) + " jittered sequences.")
                all_sequences_to_keep = [list(x) for x in set(tuple(x) for x in all_sequences_to_keep)]
                print("AFTER de-duping, there are " + str(len(all_sequences_to_keep)) + " jittered sequences.")
        ############### Else, No Jitter. Sample every Nth frame ###################
        else: # we do not want to jitter, we just want to keep frames according to a rule
            # For every frame group, we will keep the middle frame of the group.
            # If there are an even number, we will keep the left middle value (e.g., 1, 2, 3, 4 --> 2)
            #
            # start with empty sequence
            sequence_to_keep=[]
            #Loop through index groups and their counts
            for fgroup, fgcount in gcdict.items():
                # Pick the index in the middle of the group (left middle if even number)
                # 1,2,3,4,5 --> int(median)=3
                # 1,2,3,4 --> int(median)=2
                sequence_to_keep.append(fgroups.index(fgroup)+int(median(range(1,fgcount+1))))
            all_sequences_to_keep.append(sequence_to_keep)


        # initialize empty output array
        # output is a GROUP OF VIDEOS, not a single video, like the input
        output_sampled_videos=np.empty(tuple([len(all_sequences_to_keep),n_frames_to_keep] + list(framedata.shape[1:])))
        # now loop through all the sequences to keep and add those "strobed" videos to output_sampled_videos
        for seq in range(len(all_sequences_to_keep)):
            curr_video=framedata[all_sequences_to_keep[seq]]
            curr_video.shape
            output_sampled_videos[seq] = framedata[all_sequences_to_keep[seq]]

        # If we have a clas label, output it
        if class_label is not None:
            output_class_labels=np.full((len(all_sequences_to_keep)), class_label, dtype=int)
            return output_sampled_videos, output_class_labels
        # No class label, just output the sampled data
        else:
            return output_sampled_videos
