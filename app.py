import json
import logging
from connect4.video import extract_frames


def lambda_handler(event, context):
    # Assuming the video file is stored in /tmp directory within the Lambda environment

    # dummy output
    move_objects = [
        "M_42_49_x",
        "M_43_50_x",
    ]
    
    return {
        "statusCode": 200,
        "body": "\n".join(move_objects)
    }

