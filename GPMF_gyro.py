# Script to extract gopro metadata into a useful format.
# Uses a modified python-gpmf from https://github.com/rambo/python-gpmf

import gpmf.parse as gpmf_parse
from gpmf.extract import get_gpmf_payloads_from_file
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

class Extractor:
    def __init__(self, videopath = "hero5.mp4"):
        self.videopath = videopath

        self.payloads, parser = get_gpmf_payloads_from_file(videopath)

        self.parsed = []
        #print(f"GPMF payloads {self.payloads}")

        for gpmf_data, timestamps in self.payloads:
            self.parsed.append(gpmf_parse.parse_dict(gpmf_data))


        self.video_length = 0 # video length in seconds
        self.fps = 0
        self.find_video_length()

        # Parsed gyro samples
        self.gyro = [] 
        self.gyro_scal = 0
        self.num_gyro_samples = 0
        self.gyro_rate = 0 # gyro rate in Hz
        self.parsed_gyro = np.zeros((1,4)) # placeholder
        self.parse_gyro()

        self.accl = []

    def find_video_length(self):
        
        #find video length using openCV
        video = cv2.VideoCapture(self.videopath)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.video_length =  num_frames / self.fps
        print("Video length: {} s, framerate: {} FPS".format(self.video_length,self.fps))

        self.size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video.release()

    def parse_gyro(self):
        stream_start_ns = []
        first_frame_offset = -1
        shutter_speed = 1/120 # default. TODO: read from gpmf
        
        # get gyro and timestamps from gpmf
        # timestamps (TSMP) are nanoseconds time when the stream starts
        for fi, frame in enumerate(self.parsed):
            for si, stream in enumerate(frame["DEVC"]["STRM"]):
                # print(stream["STNM"], int.from_bytes(stream['STMP'], byteorder='big')) #
                if "GYRO" in stream:
                    self.gyro += stream["GYRO"]
                    stream_time_ns = int.from_bytes(stream['STMP'], byteorder='big')
                    stream_start_ns += [(stream_time_ns, stream['TSMP'])]
                    print('stream start: {} micro seconds = {}ms - samples: {}'.format(stream_time_ns, stream_time_ns / 1000, len(stream["GYRO"])))
                    # if si != 0:
                        # print('STMP: {}'.format(int.from_bytes(stream['STMP'], byteorder='big')))
                        # print('STMP: {}'.format(struct.unpack_from('<i', stream['STMP'], 4)[0]))
                    
                    # Calibration scale shouldn't change
                    self.gyro_scal = stream["SCAL"]
                    #print(self.gyro_scal) print stream name
                if first_frame_offset == -1 and "SHUT" in stream:
                    first_frame_offset = int.from_bytes(stream['STMP'], byteorder='big')
                    # print(stream['SHUT'])

                if fi == 0 and "SHUT" in stream:
                    print(f"First SHUT in stream {si} at {int.from_bytes(stream['STMP'], byteorder='big')}")
                if fi == 0 and "CORI" in stream:
                    print(f"First CORI in stream {si} at {int.from_bytes(stream['STMP'], byteorder='big')}")
        
        
        print("first frame: {} - first gyro: {} = {}".format(first_frame_offset, stream_start_ns[0][0], stream_start_ns[0][0] - first_frame_offset))

        # Convert to angular vel. vector in rad/s
        omega = np.array(self.gyro) / self.gyro_scal
        self.num_gyro_samples = omega.shape[0]

        # print('GPMF: {}'.format(omega[0]))
        # print('frame: {}'.format(self.parsed[0]['DEVC']['STRM'][1]))
        # print('stream_start_ns: {}, length: {}'.format(stream_start_ns, len(stream_start_ns)))
        # print(self.num_gyro_samples)

        self.gyro_rate = self.num_gyro_samples / self.video_length 
        print("Gyro rate: {} Hz, should be close to 200 or 400 Hz".format(self.gyro_rate))

        timestamps = []
        for i, tuple in enumerate(stream_start_ns):
            prev_tuple = stream_start_ns[i-1] if i > 0 else (0,0)
            next_tuple = stream_start_ns[i+1] if i+1 < len(stream_start_ns) else stream_start_ns[i-1]
            deltaNs = abs(next_tuple[0] - tuple[0])
            deltaSamples = abs(tuple[1] - prev_tuple[1])
            ns_per_sample = deltaNs / deltaSamples
            # print(i, deltaNs, deltaSamples, ns_per_sample, tuple)
            s_per_sample = ns_per_sample / 1000000
            print(f"next timestamps: \t {1/s_per_sample}hz \t * {deltaSamples} = {deltaNs} \t (from {tuple[0]})")
            timestamps.append(np.arange(deltaSamples) * s_per_sample + ((tuple[0] - first_frame_offset)/1000000))


        # timestamps = []
        # for i, tuple in enumerate(stream_start_ns):

        # print(timestamps[0], timestamps[1])


        # timestamps = np.concatinate([np.arange()])

        self.parsed_gyro = np.zeros((self.num_gyro_samples, 4))
        # self.parsed_gyro[:,0] = np.arange(self.num_gyro_samples) * 1/self.gyro_rate
        self.parsed_gyro[:,0] = np.concatenate(timestamps)

        # Data order for gopro gyro is (z,x,y)
        self.parsed_gyro[:,3] = omega[:,0] # z
        self.parsed_gyro[:,1] = omega[:,1] # x
        self.parsed_gyro[:,2] = omega[:,2] # y
        
    def parse_accl(self):
        for frame in self.parsed:
            for stream in frame["DEVC"]["STRM"]:
                if "ACCL" in stream:
                    #print(stream["STNM"]) # print stream name
                    self.accl += stream["ACCL"]
                    
                    # Calibration scale shouldn't change
                    self.accl_scal = stream["SCAL"]
                    #print(self.accl_scal)
        
        
        # Convert to angular vel. vector in rad/s ??
        omega = np.array(self.accl) / self.accl_scal / 9.80665
        self.num_accl_samples = omega.shape[0]

        self.accl_rate = self.num_accl_samples / self.video_length 
        print("Accl rate: {} Hz, should be close to 200 or 400 Hz".format(self.accl_rate))


        self.parsed_accl = np.zeros((self.num_accl_samples, 4))
        self.parsed_accl[:,0] = np.arange(self.num_accl_samples) * 1/self.accl_rate

        # Data order for gopro gyro is (z,x,y)
        self.parsed_accl[:,3] = omega[:,0] # z
        self.parsed_accl[:,1] = omega[:,1] # x
        self.parsed_accl[:,2] = omega[:,2] # y

    def get_gyro(self, with_timestamp = False):
        if with_timestamp:
            return self.parsed_gyro
        return self.parsed_gyro[:,1:]
    
    def get_accl(self, with_timestanp = False):
        if with_timestanp:
            return self.parsed_accl
        return self.parsed_accl[:,1:]

    def get_video_length(self):
        return self.video_length

    def has_gpmf(self, filepath):
        pass



if __name__ == "__main__":
    testing = Extractor()
    testing.get_gyro()