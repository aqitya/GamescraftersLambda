import json
import logging
from connect4.video import extract_frames

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

print(extract_frames('c4cv.mov'))


def lambda_handler(event, context):
    # Assuming the video file is stored in /tmp directory within the Lambda environment
    video_path = '/tmp/c4cv.mov'
    output_strings = extract_frames(video_path)

    # dummy output
    # output_strings = [
    #     "p=1_------X",
    #     "p=2_-O----X",
    #     "p=1_-OX---X",
    #     "p=2_XOX---X",
    # ]

    for string in output_strings:
        print(string)
    
    return {
        "statusCode": 200,
        "body": "\n".join(output_strings)
    }

